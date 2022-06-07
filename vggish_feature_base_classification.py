import numpy as np
import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

import os
import librosa

checkpoint_path = 'check_points/vggish_model.ckpt'
pca_params_path = 'check_points/vggish_pca_params.npz'

audio_path = 'C:/Users/jae/Music/hearing_loss_split_epp'

def vggish_feat_extract(x, sr):
    input_batch = vggish_input.waveform_to_examples(x, sr)
        
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
        
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME
        )
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME
        )
        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: input_batch})
        
        return embedding_batch
    
def extract_feature_from_files(category='normal'):
    min_len = 100
    max_len = 0
    vggish_feature = np.empty((0, 128))
    for folder in os.listdir(os.path.join(audio_path, category)):
        for audio_file in os.listdir(os.path.join(audio_path, category, folder)):
            if audio_file.endswith('.wav') and (audio_file.lower() == 'chapter1.wav' or audio_file.lower() == 'chapter2.wav'):
                x, sr = librosa.load(os.path.join(audio_path, category, folder, audio_file), sr=44100, mono=True)
                temp = np.array(vggish_feat_extract(x, sr))
                if audio_file.lower() == 'chapter1.wav':
                    if min_len > temp.shape[0]: min_len = temp.shape[0]
                    if max_len < temp.shape[0]: max_len = temp.shape[0]
                vggish_feature = np.append(vggish_feature, temp, axis=0)
    print(min_len, max_len)
    return vggish_feature

normal_feat = np.array(extract_feature_from_files(category='normal'))
mild_feat = np.array(extract_feature_from_files(category='mild'))
severe_feat = np.array(extract_feature_from_files(category='severe'))

print(normal_feat.shape)
print(mild_feat.shape)
print(severe_feat.shape)