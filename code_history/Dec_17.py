"""
This files defines the decoder model and trains it. Please specify your variables for this code to work.

"""

PATH_DIRECTORY_OS_DIVIDER="/"
MIDI_DATASET_DIRECTORY="/home/gordonkim/Meloson_Train/datasets"
DATASET_URL="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"


import os

os.system("pip3 install pandas")
os.system("pip3 install pretty_midi")
os.system("pip3 install matplotlib")
#os.system("pip3 install keras_nlp --no-deps")
#os.system("pip3 install keras_hub --no-deps")

# import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi

# import seaborn as sns
import tensorflow as tf
import keras

# from IPython import display
from matplotlib import pyplot as plt
from typing import Optional
import zipfile
import requests

import platform

import random

import gc

import collections


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)



max_time_shift_steps=100








filenames=[]
for root, dirs, files in os.walk(MIDI_DATASET_DIRECTORY):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                filenames.append(os.path.join(root, file))


#ensure to use tensorflow backend
os.environ["KERAS_BACKEND"] = "tensorflow"


#allow GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')

print(physical_devices)

for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

def sequence_to_midi(id_sequence, id_to_event, output_file_path):
    import pretty_midi

    # Constants
    STEPS_PER_SECOND = 100  # Must match the encoding steps per second
    VELOCITY_BINS = 32
    MAX_VELOCITY = 127

    # De-quantization function
    def dequantize_velocity(velocity_bin):
        velocity = int((velocity_bin / (VELOCITY_BINS - 1)) * MAX_VELOCITY)
        return velocity

    # Initialize PrettyMIDI object and instrument
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # Initialize variables
    current_time = 0  # in steps
    current_velocity = 64  # Default velocity
    active_notes = {}  # {pitch: (start_time_in_steps, velocity_at_note_on)}

    # Iterate over the event sequence
    for event_id in id_sequence:
        event = id_to_event[event_id]
        event_type = event['type']

        if event_type == 'note_on':
            pitch = event['pitch']
            # Check if the note is already active
            if pitch in active_notes:
                # End the previous note
                start_time, velocity = active_notes[pitch]
                end_time = current_time
                duration = end_time - start_time
                if duration > 0:
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time / STEPS_PER_SECOND,
                        end=end_time / STEPS_PER_SECOND
                    )
                    instrument.notes.append(note)
                del active_notes[pitch]
            # Start a new note with the current velocity
            active_notes[pitch] = (current_time, current_velocity)

        elif event_type == 'note_off':
            pitch = event['pitch']
            if pitch in active_notes:
                start_time, velocity = active_notes[pitch]
                end_time = current_time
                duration = end_time - start_time
                if duration > 0:
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time / STEPS_PER_SECOND,
                        end=end_time / STEPS_PER_SECOND
                    )
                    instrument.notes.append(note)
                del active_notes[pitch]
            else:
                # Handle note_off without prior note_on
                pass

        elif event_type == 'time_shift':
            steps = event['steps']
            current_time += steps

        elif event_type == 'velocity':
            # De-quantize the velocity bin back to MIDI velocity
            velocity_bin = event['value']
            current_velocity = dequantize_velocity(velocity_bin)
            current_velocity = max(0, min(current_velocity, MAX_VELOCITY))

    # After processing all events, close any remaining active notes
    for pitch, (start_time, velocity) in active_notes.items():
        end_time = current_time
        duration = end_time - start_time
        if duration > 0:
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time / STEPS_PER_SECOND,
                end=end_time / STEPS_PER_SECOND
            )
            instrument.notes.append(note)
    active_notes.clear()

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)

    # Write the PrettyMIDI object to a MIDI file
    midi.write(output_file_path)


def notes_to_events(midi_file, pitch_shift=0, velocity_shift=0, time_scale=1.0):

    """
    
    oh my beloved notes_to_events function,

    slightly misnamed, should be midi_to_events
    preprocessing function that reads a midi file and then converts it into a sequence of events
    midi_file: str, path to the midi file
    pitch_shift: int, shift in pitch
    velocity_shift: int, shift in velocity
    time_scale: float, scale in time

    pitch and velocity shifts exceeding 127 or below 0 will be clamped to 127 or 0 respectively
    both are in standard midi range, 0 to 127, and shifts are added to the original values
    keep in mind that time_scale greater than 1 will increase the note time (slower) and less than 1 will decrease the note time (faster)
    12 midi pitch values fit in a single octave

    """
    assert time_scale > 0, "Time scale must be positive"

    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    #notes = collections.defaultdict(list)

    #sort prioritizing start time and then pitch
    #sort the notes by increasing pitch
    sorted_notes = sorted(instrument.notes, key=lambda note: note.pitch)

    # Sort the notes by start time
    sorted_notes = sorted(sorted_notes, key=lambda note: note.start)

    #APPEARANTLY, we have to divide the sorting into two, otherwise it will goof up the sequence.


    # Constants for quantization
    STEPS_PER_SECOND = 100  # Adjust as needed
    MAX_SHIFT_STEPS = max_time_shift_steps  # Maximum time shift in steps, should be 100 in this code
    VELOCITY_BINS = 32
    MAX_VELOCITY = 127

    # Helper functions
    def quantize_time_augmented(t):
        scaled_time = t * time_scale
        return int(round(scaled_time * STEPS_PER_SECOND))

    def quantize_velocity_augmented(v):
        shifted_velocity = v + velocity_shift  # Apply velocity shift
        # Clamp the velocity to [0, MAX_VELOCITY]
        clamped_velocity = max(0, min(shifted_velocity, MAX_VELOCITY))
        # Quantize velocity into VELOCITY_BINS
        velocity_bin = int((clamped_velocity / MAX_VELOCITY) * (VELOCITY_BINS - 1))
        # Map back to velocity value
        return int((velocity_bin / (VELOCITY_BINS - 1)) * MAX_VELOCITY)

    # Collect all note_on and note_off events
    event_list = []
    for note in sorted_notes:
        adjusted_pitch = max(0, min(note.pitch + pitch_shift, 127)) #shifts up/down the pitch and clamps it

        start_step = quantize_time_augmented(note.start)
        end_step = quantize_time_augmented(note.end)
        velocity = quantize_velocity_augmented(note.velocity)
        event_list.append({'time': start_step, 'type': 'note_on', 'pitch': adjusted_pitch, 'velocity': velocity})
        event_list.append({'time': end_step, 'type': 'note_off', 'pitch': adjusted_pitch})

    # Sort the events by time, note_off before note_on if times are equal
    event_order = {'note_off': 0, 'note_on': 1}

    #old
    #event_list.sort(key=lambda e: (e['time'], event_order[e['type']]))
    event_list.sort(key=lambda e: (e['time'], event_order[e['type']], e['pitch']))



    # Initialize variables
    current_step = 0
    current_velocity = None  # No velocity set yet
    events = []

    for event in event_list:
        # Insert time shift if needed
        if event['time'] > current_step:
            shift_steps = event['time'] - current_step
            while shift_steps > 0:
                time_shift = min(shift_steps, MAX_SHIFT_STEPS)
                events.append({'type': 'time_shift', 'steps': time_shift})
                shift_steps -= time_shift
            current_step = event['time']

        # Handle velocity change if needed
        if event['type'] == 'note_on':
            if event['velocity'] != current_velocity:
                events.append({'type': 'velocity', 'value': event['velocity']})
                current_velocity = event['velocity']
            # Add note_on event
            events.append({'type': 'note_on', 'pitch': event['pitch']})
        elif event['type'] == 'note_off':
            # Add note_off event
            events.append({'type': 'note_off', 'pitch': event['pitch']})

    #LOGICAL ERROR FOUND: must sort the event list back into chronological order. this is not happening.

    return events


def create_event_vocabulary():
    # Constants (ensure they match those used in notes_to_events)
    NUM_PITCHES = 128  # MIDI pitches range from 0 to 127
    MAX_SHIFT_STEPS = max_time_shift_steps  # Maximum time shift steps
    VELOCITY_BINS = 32
    MAX_VELOCITY = 127

    # Initialize vocabulary dictionaries
    event_to_id = {'PAD': 0}
    id_to_event = {0: {'type': 'PAD'}}
    current_id = 1  # Start from 1, reserve 0 for padding token

    # Note-on events
    for pitch in range(NUM_PITCHES):
        event = f'note_on_{pitch}'
        event_to_id[event] = current_id
        id_to_event[current_id] = {'type': 'note_on', 'pitch': pitch}
        current_id += 1

    # Note-off events
    for pitch in range(NUM_PITCHES):
        event = f'note_off_{pitch}'
        event_to_id[event] = current_id
        id_to_event[current_id] = {'type': 'note_off', 'pitch': pitch}
        current_id += 1

    # Time-shift events
    for shift in range(1, MAX_SHIFT_STEPS + 1):
        event = f'time_shift_{shift}'
        event_to_id[event] = current_id
        id_to_event[current_id] = {'type': 'time_shift', 'steps': shift}
        current_id += 1

    # Velocity events
    velocity_values = [int((i / (VELOCITY_BINS - 1)) * MAX_VELOCITY) for i in range(VELOCITY_BINS)]
    for velocity in velocity_values:
        event = f'velocity_{velocity}'
        event_to_id[event] = current_id
        id_to_event[current_id] = {'type': 'velocity', 'value': velocity}
        current_id += 1

    return event_to_id, id_to_event



def events_to_sequence(events, event_to_id):
    NUM_PITCHES = 128  # MIDI pitches range from 0 to 127
    MAX_SHIFT_STEPS = max_time_shift_steps  # Maximum time shift steps
    VELOCITY_BINS = 32
    MAX_VELOCITY = 127
    sequence = []
    for event in events:
        if event['type'] == 'note_on':
            event_id = event_to_id[f"note_on_{event['pitch']}"]
            sequence.append(event_id)
        elif event['type'] == 'note_off':
            event_id = event_to_id[f"note_off_{event['pitch']}"]
            sequence.append(event_id)
        elif event['type'] == 'time_shift':
            # Break down large time shifts if necessary
            shift_steps = event['steps']
            while shift_steps > 0:
                shift = min(shift_steps, MAX_SHIFT_STEPS)
                event_id = event_to_id[f"time_shift_{shift}"]
                sequence.append(event_id)
                shift_steps -= shift
        elif event['type'] == 'velocity':
            event_id = event_to_id[f"velocity_{event['value']}"]
            sequence.append(event_id)
    return sequence



event_to_id, id_to_event = create_event_vocabulary()
"""
#preprocess a sample midi file
processed_events = notes_to_events(filenames[0], pitch_shift=7, velocity_shift=64, time_scale=0.75)
print(len(filenames))

for event in processed_events[:120]:
    print(event)
print(type(processed_events))
#print(id_to_event)
#print(type(id_to_event))
preprocessed_midi, _ = events_to_sequence(processed_events, event_to_id)

sequence_to_midi(preprocessed_midi, id_to_event, "/Users/gordonkim/Desktop/myStuff/ML_directory/transformers/debug_dataset/ballade4_reconstructed.mid")

#working to this point"""


#we are walking into the scary territory of creating a tf.data.Dataset processing steps

def data_generator(filenames, event_to_id, seq_length=1024, step_size=1, random_start=True, random_start_range=1024, augment_data=True):
    """
    Generator function that processes MIDI files and yields input-target pairs.

    Args:
        filenames (list): List of MIDI file paths.
        event_to_id (dict): Vocabulary mapping events to unique IDs.
        seq_length (int): Length of each input sequence.
        step_size (int): Step size for the sliding window.
        random_start (bool): Whether to start the sliding window at a random index.
        random_start_range (int): Range for random starting index.

    Yields:
        Tuple[np.ndarray, np.ndarray]: Input and target sequences.
    """
    for filename in filenames:
        try:
            # Convert MIDI to events

            if augment_data:

                pitch_shift = random.randint(-12, 12)
                velocity_shift = random.randint(-32, 32)
                time_scale = random.uniform(0.5, 2.0)
            
            else:
                pitch_shift = 0
                velocity_shift = 0
                time_scale = 1.0

            events = notes_to_events(filename, pitch_shift=pitch_shift, velocity_shift=velocity_shift, time_scale=time_scale)

            # Encode events to sequence of IDs
            sequence = events_to_sequence(events, event_to_id)

            # Convert to NumPy array
            sequence = np.array(sequence, dtype=np.int32)

            if len(sequence) <= seq_length:
                continue  # Skip sequences that are too short

            # Random starting index
            if random_start:
                max_start = min(random_start_range, len(sequence) - seq_length - 1)
                if max_start <= 0:
                    start_idx = 0
                else:
                    start_idx = np.random.randint(0, max_start)
            else:
                start_idx = 0

            # Calculate the number of sequences
            num_sequences = (len(sequence) - seq_length - start_idx) // step_size

            for i in range(num_sequences):
                input_seq = sequence[start_idx + i * step_size : start_idx + i * step_size + seq_length]
                target_seq = sequence[start_idx + i * step_size + 1 : start_idx + i * step_size + seq_length + 1]

                yield input_seq, target_seq

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue  # Skip files that cause errors

def create_tf_dataset(filenames, event_to_id, batch_size=64, seq_length=128, step_size=1, random_start=True, random_start_range=1024, augment_data=True, shuffle_buffer=1000000):
    """
    Creates a TensorFlow dataset from MIDI filenames.

    Args:
        filenames (list): List of MIDI file paths.
        event_to_id (dict): Vocabulary mapping events to unique IDs.
        batch_size (int): Batch size.
        seq_length (int): Length of each input sequence.
        step_size (int): Step size for the sliding window.
        random_start (bool): Whether to start the sliding window at a random index.
        random_start_range (int): Range for random starting index.

    Returns:
        tf.data.Dataset: TensorFlow dataset ready for training.
    """
    # Define the generator
    generator = lambda: data_generator(filenames, event_to_id, seq_length, step_size, random_start, random_start_range, augment_data=augment_data)

    # Define output signature
    output_signature = (
        tf.TensorSpec(shape=(seq_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(seq_length,), dtype=tf.int32)
    )

    # Create the dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    # Map to include padding masks
    dataset = dataset.map(
        lambda x, y: (
            {
                'token_ids': x,
                'padding_mask': tf.ones_like(x, dtype=tf.int32)
            },
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

#yeah i mean i guess it works


#a little bit scarier: creating the transformer model

layers = tf.keras.layers

def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=1024, depth=d_model) #CONTEXT SIZE

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    

class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super(CausalSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.key_dim = d_model // num_heads  # Correct key_dim per head

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=dropout_rate,

        )
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        # Apply MultiHeadAttention with causal mask
        attn_output, attn_scores = self.mha(
            query=x,
            value=x,
            key=x,
            return_attention_scores=True,
            use_causal_mask=True  # Ensures causal masking
        )
        # Cache the attention scores if needed
        self.last_attn_scores = attn_scores
        # Add & Normalize
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
    

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, d_model=d_model, dropout_rate=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def call(self, x):
        x = self.causal_self_attention(x=x)
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.causal_self_attention.last_attn_scores
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x

class Transformer(tf.keras.Model):
    def __init__(
        self, *, num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate=0.1
    ):
        super().__init__()

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate,
        )

        self.final_layer = tf.keras.layers.Dense(input_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        token_ids, padding_mask = inputs['token_ids'], inputs['padding_mask']

        x = self.decoder(token_ids)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


#use TPU strat if available


tpu_device = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(tpu_device)
tf.tpu.experimental.initialize_tpu_system(tpu_device)
strategy = tf.distribute.TPUStrategy(tpu_device)



#CHECK HERE

make_new_model=False
change_compiler=True

learning_rate_forthemodel=3e-5

with strategy.scope():
        
    if make_new_model:

        gpt2_causal_lm = Transformer(
            num_layers=12,
            d_model=768,
            num_heads=12,
            dff=3072,
            input_vocab_size=len(event_to_id),
            dropout_rate=0.1
        )

    else:
        gpt2_causal_lm = tf.keras.models.load_model("meloson_gpt2.keras", custom_objects={"PositionalEmbedding": PositionalEmbedding, "CausalSelfAttention": CausalSelfAttention, "FeedForward": FeedForward, "DecoderLayer": DecoderLayer, "Decoder": Decoder, "Transformer": Transformer}, compile=False)

    if change_compiler:


        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate_forthemodel, weight_decay=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        gpt2_causal_lm.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    else:
        gpt2_causal_lm.compile()

    #dummy data run
    dummy_input = {
        'token_ids': tf.constant(tf.ones((1, 1024), dtype=tf.int32)),
        'padding_mask': tf.constant(tf.ones((1, 1024), dtype=tf.int32)),
    }

    gpt2_causal_lm(dummy_input)


    #train da model!!!


    epochs=999999

    for i in range(epochs):


        random.seed(np.random.randint(0, 10000))

        random.shuffle(filenames)

        train_ratio=0.9
        split_index = int(len(filenames) * train_ratio)

        train_filenames = filenames[:split_index]
        test_filenames = filenames[split_index:]

        print(f"Total files: {len(filenames)}")
        print(f"Training files: {len(train_filenames)}")
        print(f"Testing files: {len(test_filenames)}")

        train_dataset = create_tf_dataset(
            filenames=train_filenames,
            event_to_id=event_to_id,
            batch_size=128,
            seq_length=1024,
            step_size=256,
            random_start=True,
            random_start_range=1024,
            augment_data=True,
            shuffle_buffer=10000
        )

        test_dataset = create_tf_dataset(
            filenames=test_filenames,
            event_to_id=event_to_id,
            batch_size=128,
            seq_length=1024,
            step_size=256,
            random_start=True,
            random_start_range=1024,
            augment_data=False,
            shuffle_buffer=1000
        )

        #for i in range(epochs):

        for batch in train_dataset.take(1):
            inputs, targets = batch
            print(inputs)
            print(targets)
            break

        gpt2_causal_lm.summary()


        gpt2_causal_lm.fit(train_dataset, epochs=1, validation_data=test_dataset, validation_steps=1000)
        gpt2_causal_lm.save("meloson_gpt2.keras")



