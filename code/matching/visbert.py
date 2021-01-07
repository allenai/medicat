import sys

import torch

from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
from transformers.modeling_bert import BertEmbeddings, BertPooler, BertLayer, BertLayerNorm, BertModel, gelu, BertOnlyMLMHead
from transformers.file_utils import add_start_docstrings_to_callable

class BertEncoder(torch.nn.Module):
    def __init__(self, config, visual_start_layer):
        super().__init__()
        self.output_attentions = False # config.output_attentions
        self.output_hidden_states = True # config.output_hidden_states
        self.visual_start_layer = visual_start_layer
        self.config = config
        self.layer = torch.nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        visual_states=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        exit_layer=None
    ):
        all_hidden_states = ()
        all_attentions = ()
        if head_mask is None:
            head_mask = [None]*self.config.num_hidden_layers
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i == self.visual_start_layer and visual_states is not None:
                if len(visual_states.shape) == 3:
                    attention_mask = torch.cat((attention_mask, torch.zeros_like(attention_mask[:,:,:,:1].repeat(1, 1, 1, visual_states.shape[1]))), dim=-1)
                    attention_mask = torch.cat((attention_mask, torch.zeros_like(attention_mask[:,:,:1,:].repeat(1, 1, visual_states.shape[1], 1))), dim=-2)
                    hidden_states = torch.cat((hidden_states, visual_states), dim=1)
                else:
                    attention_mask = torch.cat((attention_mask, torch.zeros_like(attention_mask[:,:,:,:1])), dim=-1)
                    hidden_states = torch.cat((hidden_states, visual_states.unsqueeze(1)), dim=1)
            layer_outputs = layer_module(
                hidden_states=hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if exit_layer == i:
                break

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class VisBertModel(BertPreTrainedModel):
    def __init__(self, config, visual_feat_size, visual_start_layer, num_visual_positions, use_pos_embedding=False, no_encoder_inputs=False, append_to_encoder_states=False):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.use_pos_embedding = use_pos_embedding
        if use_pos_embedding:
            self.visual_pos_embeddings = torch.nn.Embedding(num_visual_positions, config.hidden_size)
        else:
            self.visual_pos_embeddings = torch.nn.Linear(4, config.hidden_size)
        self.visual_feat_size = visual_feat_size
        self.visual_start_layer = visual_start_layer
        self.num_visual_positions = num_visual_positions
        self.visual_feat_projection = torch.nn.Linear(visual_feat_size, config.hidden_size)
        self.encoder = BertEncoder(config, visual_start_layer)
        self.pooler = BertPooler(config)
        self.dropout_layer = torch.nn.Dropout(p=self.config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)
        self.no_encoder_inputs = no_encoder_inputs
        self.append_to_encoder_states = append_to_encoder_states

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            visual_inputs=None, visual_attention_mask=None, exit_layer=None, pool=True,
            encoder_hidden_states=None, encoder_attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        """input_shape = input_ids.size()
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, self.device)"""

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        if self.no_encoder_inputs:
            encoder_hidden_states = None
            encoder_extended_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Visual input projection
        if visual_inputs:
            pos, img_feats = visual_inputs
            if self.use_pos_embedding:
                visual_embeddings = self.dropout_layer(self.visual_pos_embeddings(pos)+self.visual_feat_projection(img_feats))
            else:
                visual_embeddings = self.dropout_layer(self.visual_pos_embeddings(pos)+self.visual_feat_projection(img_feats))
            if self.append_to_encoder_states:
                assert self.use_pos_embedding
                visual_attention_mask = torch.zeros_like(visual_embeddings[:,:,0]).float()
                visual_attention_mask = visual_attention_mask[:,None,None,:]
                if self.no_encoder_inputs:
                    encoder_hidden_states = visual_embeddings
                    encoder_extended_attention_mask = visual_attention_mask
                else:
                    encoder_hidden_states = torch.cat((encoder_hidden_states, visual_embeddings), dim=1)
                    encoder_extended_attention_mask = torch.cat((encoder_extended_attention_mask, visual_attention_mask), dim=-1)
                visual_embeddings = None
        else:
            visual_embeddings = None

        # Run LXRT backbone
        outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            visual_states=visual_embeddings,
            exit_layer=exit_layer,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask
        )
        if pool:
            pooled_output = self.pooler(outputs[0])
        else:
            pooled_output = outputs[0][:,0,:]

        return outputs[0], pooled_output
