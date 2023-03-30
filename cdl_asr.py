import os
import numpy as np
import pandas as pd
import nemo.collections.asr as nemo_asr
import copy
import torch

def softmax(logits):
    '''softmax used by Nemo'''
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


def load_default_lm(lm_path):
    '''make sure that there is a language model at the lm path and download a default if it isn't there yet'''
    if os.path.exists(lm_path):
        print('LM model already exists at '+lm_path)
    else:
        print('LM model not found at '+lm_path+'; Downloading and preparing...')

        lm_gzip_path = '3-gram.pruned.1e-7.arpa.gz'
        if not os.path.exists(lm_gzip_path):
            print('Downloading pruned 3-gram model.')
            lm_url = 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
            lm_gzip_path = wget.download(lm_url)
            print('Downloaded the 3-gram language model.')
        else:
            print('Pruned .arpa.gz already exists.')

        uppercase_lm_path = '3-gram.pruned.1e-7.arpa'
        if not os.path.exists(uppercase_lm_path):
            with gzip.open(lm_gzip_path, 'rb') as f_zipped:
                with open(uppercase_lm_path, 'wb') as f_unzipped:
                    shutil.copyfileobj(f_zipped, f_unzipped)
            print('Unzipped the 3-gram language model.')
        else:
            print('Unzipped .arpa already exists.')

        lm_path = 'lowercase_3-gram.pruned.1e-7.arpa'
        if not os.path.exists(lm_path):
            with open(uppercase_lm_path, 'r') as f_upper:
                with open(lm_path, 'w') as f_lower:
                    for line in f_upper:
                        f_lower.write(line.lower())
        print('Converted language model file to lowercase. Ready for use!')


def transcribe_with_ngram_model(files, lm_path, asr_model):
    '''one or more audio files using an ngram model specified by lm_path'''

    transcript = asr_model.transcribe(paths2audio_files=files)[0]
    print(f'Transcript: "{transcript}"')

    # inference without a language model
    logits = asr_model.transcribe(files, logprobs=True)[0]
    probs = softmax(logits)

    beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
        vocab=list(asr_model.decoder.vocabulary),
        beam_width=256,
        alpha=10, beta=1.5,
        lm_path=lm_path,
        num_cpus=max(os.cpu_count(), 1),
    input_tensor=False)

    beam_results_tuple = beam_search_lm.forward(log_probs = np.expand_dims(probs, axis=0), log_probs_length=None)[0]
    beam_results_df = pd.DataFrame({'hypothesis':[x[1] for x in beam_results_tuple] ,'ngram_prob':[-1. * x[0] for x in beam_results_tuple]})
    beam_results_df = beam_results_df.sort_values(by=['ngram_prob'])
    return(beam_results_df)

def mask_each(text, tokenizer):
    '''produce a set of texts with each word masked in turn. Returns a list of tupes with (masked utterance, true word)'''
    tokenized_text = tokenizer.tokenize('[CLS] '+text+'. [SEP]')    
    masked_utts = [ ]
    for i in range(1,len(tokenized_text)-1):
        masked_utt = copy.copy(tokenized_text)
        true_word = masked_utt[i] 
        masked_utt[i] = '[MASK]'
        masked_utts.append((masked_utt, true_word))
    return(masked_utts)


def compute_prob_for_one_mask(sequence_with_one_mask_and_true_word, model, vocab, tokenizer, return_type='prob'):
    '''get the probability of the true word in the mask position from a BERT model'''

    indexed_tokens, true_word = sequence_with_one_mask_and_true_word
    indexed_tokens = tokenizer.convert_tokens_to_ids(indexed_tokens)

    masked_index = indexed_tokens.index(tokenizer.convert_tokens_to_ids("[MASK]"))
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    prior_probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)

    if return_type == 'full': 
        word_predictions  = pd.DataFrame({'prior_prob': prior_probs.detach().cpu(), 'word':vocab})
        word_predictions = word_predictions.sort_values(by='prior_prob', ascending=False)    
        word_predictions['prior_rank'] = range(word_predictions.shape[0])

        rdict = {
            'prior_prob_continuations': word_predictions.loc[word_predictions.word == true_word],
            'prior_prob_all': prior_probs,
            'word_predictions': word_predictions,
            'true_word': true_word
        }
        return(rdict)

    elif return_type == 'prob': 
        prob = prior_probs.detach().cpu().numpy()[tokenizer.convert_tokens_to_ids(true_word)]
        return(prob)
    else:
        raise ValueError('Return type not recognized')
    
def compute_prob_for_all_masks(hypothesis, bertMaskedLM, vocab, tokenizer,  return_type='prob'):   
    '''Complute the pseudo log likelihood for a hypothesized utterance interpretation''' 
    all_masks = mask_each(hypothesis, tokenizer)
    if (return_type == 'full'):
        test = [compute_prob_for_one_mask(x, bertMaskedLM, vocab, tokenizer, return_type) for x in  all_masks]
        for item in test:    
            print('True token: '+ item['true_word'] +'; Predicted token: '+ item['word_predictions'].iloc[0]['word']) 
        return(test)
    elif (return_type == 'prob'):
        probs = [compute_prob_for_one_mask(x, bertMaskedLM, vocab, tokenizer,return_type) for x in  all_masks]
        return(np.sum(-1. * np.log(probs)))


def compute_prob_for_all_hypotheses(all_hypotheses, bertMaskedLM, vocab, tokenizer):
    '''Complute the pseudo log likelihoods for all hypothesized utterance interpretations''' 

    all_hypotheses['bert_prob'] = [compute_prob_for_all_masks(hypothesis, bertMaskedLM, vocab, tokenizer, return_type='prob') for hypothesis in all_hypotheses.hypothesis]
    
    return(all_hypotheses)


def transcribe_with_neural_rescoring(files, lm_path, asr_model, bertMaskedLM, vocab, tokenizer, num_hypotheses, alpha, rescore=True):
    all_hypotheses = transcribe_with_ngram_model(files, lm_path, asr_model)
    all_hypotheses = all_hypotheses.sort_values(by=['ngram_prob'])
    print('Best hypothesis after ngram: '+all_hypotheses.iloc[0].hypothesis)
    
    if rescore:
        selected_hypotheses = all_hypotheses[0:num_hypotheses]
        rescored_hypotheses = compute_prob_for_all_hypotheses(selected_hypotheses, bertMaskedLM, vocab, tokenizer)
        rescored_hypotheses['interpolated_prob'] = (1. - alpha) * rescored_hypotheses['ngram_prob'] + alpha * rescored_hypotheses['bert_prob']
        rescored_hypotheses = rescored_hypotheses.sort_values(by=['interpolated_prob'])
        return(rescored_hypotheses)
    else:
        return(all_hypotheses)
