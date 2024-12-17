

import os
import tensorflow as tf
#check if tensorflow is 2.16.1, if not, down/upgrade to it

"""
if tf.__version__ != "2.18.0":
    !pip install tensorflow==2.18.0"""

#!pip install tensorflow-gcs-config
#!pip install keras_nlp --no-deps
#!pip install keras_hub --no-deps
#!pip install pretty_midi

"""os.system("pip install pandas")
os.system("pip install pretty_midi")
os.system("pip install matplotlib")
os.system("pip install keras_nlp --no-deps")
os.system("pip install keras_hub --no-deps")"""



import collections
import datetime

# import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import platform

# import seaborn as sns
import tensorflow as tf

# from IPython import display
from matplotlib import pyplot as plt
from typing import Optional

import os
import random


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000

if os.name=="nt":
    PATH_DIRECTORY_OS_DIVIDER="\\"
    MIDI_DATASET_DIRECTORY="C:\\Users\\037co\\OneDrive\\Desktop\\MyStuff\\data\\LLModels\\muRNNtrain\\datasets"
    batch_size=512
    epochs=1024
elif(platform.system() == "Darwin"):
    PATH_DIRECTORY_OS_DIVIDER="/"
    MIDI_DATASET_DIRECTORY="//Users/gordonkim/Desktop/myStuff/ML_directory/transformers/datasets"
    epochs=8
    batch_size=64
    model_path="/Users/gordonkim/Desktop/decoder.keras"
    generated_files_directory="/Users/gordonkim/Desktop/transformergenerated"
    #import os
    #os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
else:
    #this vastaitrain is for vasti.ai container, crazy gpus are gonna be used
    #my wallet is gonna be consumed by the black hole
    #worth it right? 😭

    """
    PATH_DIRECTORY_OS_DIVIDER="/"
    MIDI_DATASET_DIRECTORY="/content/drive/MyDrive/ML_directory/midi_datasets"
    DATASET_URL="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
    ZIP_FILE_PATH = os.path.join(MIDI_DATASET_DIRECTORY, "maestro-v2.0.0-midi.zip")
    generated_files_directory="/content/drive/MyDrive/ML_directory/generated_midis"
    model_path="/content/drive/MyDrive/ML_directory/decoder.keras"
    """

    #vastai config:
    """
    PATH_DIRECTORY_OS_DIVIDER="/"
    MIDI_DATASET_DIRECTORY="/root/vastai/maestro-v2.0.0"
    DATASET_URL="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
    ZIP_FILE_PATH = os.path.join("/root/vastai/", "maestro-v2.0.0-midi.zip")"""

    #for local PC

    PATH_DIRECTORY_OS_DIVIDER="/"
    MIDI_DATASET_DIRECTORY="/home/gordon/Desktop/vastai/fun_datasets"
    DATASET_URL="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
    ZIP_FILE_PATH = os.path.join(MIDI_DATASET_DIRECTORY, "maestro-v2.0.0-midi.zip")
    generated_files_directory="/home/gordon/Desktop/vastai/generated_midis/generated_midis"
    model_path="/home/gordon/Desktop/vastai/satisfied_models/meloson_gpt2_DEC7_PART1_.keras"



    epochs=8
    batch_size=64
    sequence_length=256
    DATSET_FILE_SIZE=400

download_midi_files=False

filenames=[]

max_time_shift_steps=100

for root, dirs, files in os.walk(MIDI_DATASET_DIRECTORY):
    for file in files:
        if file.endswith(".mid") or file.endswith(".midi"):
            filenames.append(os.path.join(root, file))

print("Number of files:", len(filenames))



import pretty_midi

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



def notes_to_midi(
    notes: pd.DataFrame,
    out_file: str,
    instrument_name: str,
) -> pretty_midi.PrettyMIDI:

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note["step"])
        end = float(start + note["duration"])
        note = pretty_midi.Note(
            velocity=int(note["velocity"]),
            pitch=int(note["pitch"]),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm

time_vocab_size=100
def roundToNearest(x,base=1/100):

    #rounded=round(x/base)
    #divide by every 10 milliseconds
    reciprocal_base = 100  # Precompute the reciprocal of base
    rounded = int(round(1000*x*base))  # Use integer division for rounding

    return rounded


def roundToNearestVelocity(x,base=1/4):

    reciprocal_base = 4  # Precompute the reciprocal of base
    rounded = int(round(x*base))  # Use integer division for rounding

    if(rounded<32): #smaller than maximum velocity
        return rounded
    else:
        return 32

print(roundToNearestVelocity(0))




def midi_to_notes(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    #sort the notes by increasing pitch
    sorted_notes = sorted(instrument.notes, key=lambda note: note.pitch)

    # Sort the notes by start time
    sorted_notes = sorted(sorted_notes, key=lambda note: note.start)


    prev_start = sorted_notes[0].start


    for note in sorted_notes:
        start = roundToNearest(note.start)
        end = roundToNearest(note.end)
        notes["pitch"].append(note.pitch)
        notes["start"].append(start)
        notes["end"].append(end)

        step=start-prev_start
        duration=end-start

        notes["step"].append(step)
        notes["duration"].append(duration)
        notes["velocity"].append(roundToNearestVelocity(note.velocity))

        prev_start = start





    #return notes
    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})



def notes_to_events(midi_file):

    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    #notes = collections.defaultdict(list)

    #sort the notes by increasing pitch
    sorted_notes = sorted(instrument.notes, key=lambda note: note.pitch)

    # Sort the notes by start time
    sorted_notes = sorted(sorted_notes, key=lambda note: note.start)


    # Constants for quantization
    STEPS_PER_SECOND = 100  # Adjust as needed
    MAX_SHIFT_STEPS = max_time_shift_steps  # Maximum time shift in steps #updated from 100 to 500
    VELOCITY_BINS = 32
    MAX_VELOCITY = 127

    # Helper functions
    def quantize_time(t):
        return int(round(t * STEPS_PER_SECOND))

    def quantize_velocity(velocity):
        # Quantize velocity into VELOCITY_BINS bins
        velocity_bin = int((velocity / MAX_VELOCITY) * (VELOCITY_BINS - 1))
        # Map back to velocity value
        return int((velocity_bin / (VELOCITY_BINS - 1)) * MAX_VELOCITY)

    # Collect all note_on and note_off events
    event_list = []
    for note in sorted_notes:
        start_step = quantize_time(note.start)
        end_step = quantize_time(note.end)
        velocity = quantize_velocity(note.velocity)
        event_list.append({'time': start_step, 'type': 'note_on', 'pitch': note.pitch, 'velocity': velocity})
        event_list.append({'time': end_step, 'type': 'note_off', 'pitch': note.pitch})

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
    MAX_SHIFT_STEPS = max_time_shift_steps  # Maximum time shift steps #updated from 100 to 500
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


from keras_nlp.samplers import Sampler


class MyCustomSampler(Sampler):
    def __init__(self, event_to_id, id_to_event, temperature=1.0):
        super().__init__()
        self.event_to_id = event_to_id
        self.id_to_event = id_to_event
        self.temperature = temperature
        self.active_notes = {}  # {pitch: start_time}
        self.current_time = 0  # in timesteps

    def __call__(self, logits, step, token_ids, **kwargs):
        # Get the last generated token
        last_token_id = token_ids[-1]
        last_event = self.id_to_event[last_token_id]

        # Update active_notes and current_time based on the last event
        self.update_state(last_event)

        # Adjust logits according to your constraints
        adjusted_logits = self.adjust_logits(logits[-1])

        # Apply temperature
        adjusted_logits = adjusted_logits / self.temperature

        # Sample the next token
        predicted_id = tf.random.categorical([adjusted_logits], num_samples=1)[0, 0]
        return predicted_id

    def update_state(self, event):
        event_type = event['type']
        if event_type == 'note_on':
            pitch = event['pitch']
            self.active_notes[pitch] = self.current_time
        elif event_type == 'note_off':
            pitch = event['pitch']
            self.active_notes.pop(pitch, None)
        elif event_type == 'time_shift':
            steps = event['steps']
            self.current_time += steps

    def adjust_logits(self, logits):
        import numpy as np
        logits_np = logits.numpy()

        overdue_threshold = 500  # Adjust as needed

        # Check for overdue notes
        notes_to_close = [pitch for pitch, start_time in self.active_notes.items()
                          if self.current_time - start_time >= overdue_threshold]

        if notes_to_close:
            # Force note_off events for overdue notes
            logits_np[:] = -1e9  # Zero out all probabilities
            for pitch in notes_to_close:
                note_off_event = f"note_off_{pitch}"
                event_id = self.event_to_id.get(note_off_event)
                if event_id is not None:
                    logits_np[event_id] = logits[event_id].numpy()
        else:
            # Build the list of invalid events
            invalid_event_ids = []
            for event_id_loop, event_loop in self.id_to_event.items():
                event_type_loop = event_loop['type']
                if event_type_loop == 'note_on':
                    pitch_loop = event_loop['pitch']
                    if pitch_loop in self.active_notes:
                        invalid_event_ids.append(event_id_loop)
                elif event_type_loop == 'note_off':
                    pitch_loop = event_loop['pitch']
                    if pitch_loop not in self.active_notes:
                        invalid_event_ids.append(event_id_loop)

            logits_np[invalid_event_ids] = -1e9

        return tf.convert_to_tensor(logits_np)

"""
def generate_sequence(model, start_sequence, event_to_id, id_to_event,
                      output_midi_path, num_generate=5000, temperature=1.0, sequence_length=256):
    # Prepare the input sequence

    input_sequence = start_sequence.copy()

    inputs = {
        "token_ids": tf.convert_to_tensor([input_sequence], dtype=tf.int32),
        "padding_mask": tf.ones_like([input_sequence], dtype=tf.int32)
    }

    # Build the custom sampler
    sampler = MyCustomSampler(event_to_id, id_to_event, temperature)

    # Generate the sequence
    generated_sequence = model.generate(
        inputs=inputs,
        max_length=num_generate,
        stop_token_ids=None,
        #sampler=sampler,
    )

    # Flatten the generated_sequence and convert to list
    print(type(generated_sequence))
    #print(generated_sequence.shape)
    print(generated_sequence)
    generated_sequence = generated_sequence["token_ids"].tolist()[0]
    print(generated_sequence)
    print(len(generated_sequence))

    # Truncate the sequence to the desired length
    if len(generated_sequence) > sequence_length:
        generated_sequence = generated_sequence[-sequence_length:]

    # Convert the generated sequence into a MIDI file
    sequence_to_midi(generated_sequence, id_to_event, output_midi_path)

    return generated_sequence

"""

def generate_sequence(model, start_sequence, event_to_id, id_to_event, output_midi_path, num_generate=1000, temperature=1.0, sequence_length=256):
    import tensorflow as tf
    import numpy as np

    # Prepare the input sequence
    input_sequence = start_sequence.copy()

    #empty
    generated_sequence = []

    # Initialize active_notes as a dictionary to store start times
    active_notes = {}  # {pitch: start_time}

    # Initialize current_time
    current_time = 0  # in timesteps

    # Initialize last_event
    last_event = None

    # Process the start_sequence to update active_notes and current_time
    for event_id in start_sequence:
        event = id_to_event[event_id]
        event_type = event['type']

        if event_type == 'note_on':
            pitch = event['pitch']
            active_notes[pitch] = current_time  # Store the start time
        elif event_type == 'note_off':
            pitch = event['pitch']
            active_notes.pop(pitch, None)
        elif event_type == 'time_shift':
            steps = event['steps']
            current_time += steps
        elif event_type == 'velocity':
            pass  # No action needed for velocity

    # Generate num_generate IDs
    for _ in range(num_generate):
        # Prepare the input tensor

        print("Inference step: "+ str(_), " out of " + str(num_generate))
        input_sequence_tensor = tf.expand_dims(input_sequence, 0)  # Shape: (1, seq_length)

        padding_mask=tf.ones_like(input_sequence_tensor)

        inputs = {
        "token_ids": input_sequence_tensor,
        "padding_mask": padding_mask
    }

        # Get predictions
        predictions = model.predict(inputs)  # Shape: (1, vocab_size)

        #get the last prediction

        print(type(predictions))


        #remove batch dim
        predictions = tf.squeeze(predictions, axis=0)  # Shape: (seq_len,vocab_size)
        predictions = predictions[-1, :]  # Shape: (vocab_size,)
        print(predictions.shape)

        #get the last prediction only
        #predictions = predictions[-1]  # Shape: (vocab_size,)
        #print(predictions.shape)

        # Apply temperature
        predictions = predictions / temperature

        # Convert predictions to numpy array
        #predictions_np = predictions.numpy() #i'm pretty sure predictions is already a numpy array
        predictions_np = predictions.numpy()

        overdue_threshold = 500  # 100 timesteps = 1 second

        # Check for overdue notes
        notes_to_close = [pitch for pitch, start_time in active_notes.items() if current_time - start_time >= overdue_threshold]

        if notes_to_close:
            # Force note_off events for overdue notes
            # Set all logits to a large negative number
            predictions_np[:] = -1e9  # Zero out all probabilities

            # Set logits for note_off events of overdue pitches to their original values
            for pitch in notes_to_close:
                note_off_event = f"note_off_{pitch}"
                event_id = event_to_id.get(note_off_event)
                if event_id is not None:
                    predictions_np[event_id] = predictions[event_id].numpy()  # Restore original logit
        else:
            # Adjust the logits to avoid invalid events

            # Build the list of invalid events
            invalid_event_ids = []
            for event_id_loop, event_loop in id_to_event.items():
                event_type_loop = event_loop['type']
                if event_type_loop == 'note_on':
                    pitch_loop = event_loop['pitch']
                    if pitch_loop in active_notes:
                        # note_on for already active note is invalid
                        invalid_event_ids.append(event_id_loop)
                elif event_type_loop == 'note_off':
                    pitch_loop = event_loop['pitch']
                    if pitch_loop not in active_notes:
                        # note_off for inactive note is invalid
                        invalid_event_ids.append(event_id_loop)
                # Add checks for other invalid events if necessary

            # Set logits of invalid events to a large negative number
            predictions_np[invalid_event_ids] = -1e9  # Effectively zero probability after softmax

        # Convert back to tensor
        adjusted_predictions = tf.convert_to_tensor(predictions_np)

        # Sample from the distribution
        predicted_id = tf.random.categorical([adjusted_predictions], num_samples=1)[0, 0].numpy()

        # Get the event corresponding to predicted_id
        event = id_to_event[predicted_id]
        event_type = event['type']

        # Update active_notes and current_time based on the predicted event
        if event_type == 'note_on':
            pitch = event['pitch']
            active_notes[pitch] = current_time  # Store the start time
        elif event_type == 'note_off':
            pitch = event['pitch']
            active_notes.pop(pitch, None)
        elif event_type == 'time_shift':
            steps = event['steps']
            current_time += steps
        elif event_type == 'velocity':
            pass  # No action needed for velocity

        # Append the predicted ID to the generated sequence
        generated_sequence.append(predicted_id)

        # Update the input sequence
        input_sequence.append(predicted_id)

        # Truncate the input sequence to the last seq_length events
        if len(input_sequence) > sequence_length:
            input_sequence = input_sequence[-sequence_length:]

        # Update last_event
        last_event = event

    # Convert the generated sequence into a MIDI file using the existing function
    sequence_to_midi(generated_sequence, id_to_event, output_midi_path)

    return generated_sequence

def generate_sequence(model, start_sequence, event_to_id, id_to_event, output_midi_path, num_generate=1000, temperature=1.0, sequence_length=256):

    # Prepare the input sequence
    input_sequence = start_sequence.copy()

    #empty option (only use for pure  model generated)
    generated_sequence = []

    #includes starter sequence
    #generated_sequence = start_sequence.copy()

    # Initialize active_notes as a dictionary to store start times
    active_notes = {}  # {pitch: start_time}

    # Initialize current_time
    current_time = 0  # in timesteps

    # Initialize last_event
    last_event = None

    # Process the start_sequence to update active_notes and current_time
    for event_id in start_sequence:
        event = id_to_event[event_id]
        event_type = event['type']

        if event_type == 'note_on':
            pitch = event['pitch']
            active_notes[pitch] = current_time  # Store the start time
        elif event_type == 'note_off':
            pitch = event['pitch']
            active_notes.pop(pitch, None)
        elif event_type == 'time_shift':
            steps = event['steps']
            current_time += steps
        elif event_type == 'velocity':
            pass  # No action needed for velocity

    # Generate num_generate IDs
    for _ in range(num_generate):
        # Prepare the input tensor

        print("Inference step: " + str(_) + " out of " + str(num_generate))

        input_sequence_tensor = tf.expand_dims(input_sequence, 0)  # Shape: (1, seq_length)

        padding_mask=tf.ones_like(input_sequence_tensor)

        inputs = {
        "token_ids": input_sequence_tensor,
        "padding_mask": padding_mask
    }

        # Get predictions
        predictions = model.predict(inputs)  # Shape: (1, vocab_size)

        #get the last prediction

        print(type(predictions))


        #remove batch dim
        predictions = tf.squeeze(predictions, axis=0)  # Shape: (seq_len,vocab_size)
        predictions = predictions[-1, :]  # Shape: (vocab_size,)
        print(predictions.shape)

        #get the last prediction only
        #predictions = predictions[-1]  # Shape: (vocab_size,)
        #print(predictions.shape)

        # Apply temperature
        predictions = predictions / temperature

        # Convert predictions to numpy array
        #predictions_np = predictions.numpy() #i'm pretty sure predictions is already a numpy array
        predictions_np = predictions.numpy()

        overdue_threshold = 500  # 100 timesteps = 1 second

        # Check for overdue notes
        notes_to_close = [pitch for pitch, start_time in active_notes.items() if current_time - start_time >= overdue_threshold]

        if notes_to_close:
            # Force note_off events for overdue notes
            # Set all logits to a large negative number
            predictions_np[:] = -1e9  # Zero out all probabilities

            # Set logits for note_off events of overdue pitches to their original values
            for pitch in notes_to_close:
                note_off_event = f"note_off_{pitch}"
                event_id = event_to_id.get(note_off_event)
                if event_id is not None:
                    predictions_np[event_id] = predictions[event_id].numpy()  # Restore original logit
        else:
            # Adjust the logits to avoid invalid events

            # Build the list of invalid events
            invalid_event_ids = []
            for event_id_loop, event_loop in id_to_event.items():
                event_type_loop = event_loop['type']
                if event_type_loop == 'note_on':
                    pitch_loop = event_loop['pitch']
                    if pitch_loop in active_notes:
                        # note_on for already active note is invalid
                        invalid_event_ids.append(event_id_loop)
                elif event_type_loop == 'note_off':
                    pitch_loop = event_loop['pitch']
                    if pitch_loop not in active_notes:
                        # note_off for inactive note is invalid
                        invalid_event_ids.append(event_id_loop)
                # Add checks for other invalid events if necessary

            # Set logits of invalid events to a large negative number
            predictions_np[invalid_event_ids] = -1e9  # Effectively zero probability after softmax

        # Convert back to tensor
        adjusted_predictions = tf.convert_to_tensor(predictions_np)

        # Sample from the distribution
        predicted_id = tf.random.categorical([adjusted_predictions], num_samples=1)[0, 0].numpy()

        # Get the event corresponding to predicted_id
        event = id_to_event[predicted_id]
        event_type = event['type']

        # Update active_notes and current_time based on the predicted event
        if event_type == 'note_on':
            pitch = event['pitch']
            active_notes[pitch] = current_time  # Store the start time
        elif event_type == 'note_off':
            pitch = event['pitch']
            active_notes.pop(pitch, None)
        elif event_type == 'time_shift':
            steps = event['steps']
            current_time += steps
        elif event_type == 'velocity':
            pass  # No action needed for velocity

        # Append the predicted ID to the generated sequence
        generated_sequence.append(predicted_id)

        # Update the input sequence
        input_sequence.append(predicted_id)

        # Truncate the input sequence to the last seq_length events
        if len(input_sequence) > sequence_length:
            input_sequence = input_sequence[-sequence_length:]

        # Update last_event
        last_event = event

    # Convert the generated sequence into a MIDI file using the existing function
    sequence_to_midi(generated_sequence, id_to_event, output_midi_path)

    return generated_sequence




def detect_and_set_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU")
    except ValueError:
        gpus = tf.config.experimental.list_logical_devices("GPU")
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            print("Running on GPU")
        else:
            strategy = tf.distribute.get_strategy()
            print("Running on CPU")
    return strategy

strategy = detect_and_set_strategy()


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


#use strategy.scope() to inference using the model

if True:

    model=tf.keras.models.load_model(model_path, custom_objects={"PositionalEmbedding": PositionalEmbedding, "CausalSelfAttention": CausalSelfAttention, "FeedForward": FeedForward, "DecoderLayer": DecoderLayer, "Decoder": Decoder, "Transformer": Transformer}, compile=False)

    random.shuffle(filenames)


    model.summary()

    event_to_id, id_to_event = create_event_vocabulary()




    #D major scale
    #start_sequence=[event_to_id["note_on_62"], event_to_id["time_shift_10"], event_to_id["note_off_62"], event_to_id["note_on_64"], event_to_id["time_shift_10"], event_to_id["note_off_64"], event_to_id["note_on_66"], event_to_id["time_shift_10"], event_to_id["note_off_66"], event_to_id["note_on_67"], event_to_id["time_shift_10"], event_to_id["note_off_67"], event_to_id["note_on_69"], event_to_id["time_shift_10"], event_to_id["note_off_69"], event_to_id["note_on_71"], event_to_id["time_shift_10"], event_to_id["note_off_71"], event_to_id["note_on_73"], event_to_id["time_shift_10"], event_to_id["note_off_73"], event_to_id["note_on_74"], event_to_id["time_shift_10"], event_to_id["note_off_74"], event_to_id["note_on_76"], event_to_id["time_shift_10"], event_to_id["note_off_76"], event_to_id["note_on_78"], event_to_id["time_shift_10"], event_to_id["note_off_78"], event_to_id["note_on_79"], event_to_id["time_shift_10"], event_to_id["note_off_79"], event_to_id["note_on_81"], event_to_id["time_shift_10"], event_to_id["note_off_81"], event_to_id["note_on_83"], event_to_id["time_shift_10"], event_to_id["note_off_83"], event_to_id["note_on_84"], event_to_id["time_shift_10"], event_to_id["note_off_84"], event_to_id["note_on_86"], event_to_id["time_shift_10"], event_to_id["note_off_86"], event_to_id["note_on_88"], event_to_id["time_shift_10"], event_to_id["note_off_88"], event_to_id["note_on_90"], event_to_id["time_shift_10"], event_to_id["note_off_90"], event_to_id["note_on_91"], event_to_id["time_shift_10"], event_to_id["note_off_91"], event_to_id["note_on_93"], event_to_id["time_shift_10"]]


    """#print them back into events for debugging
    start_sequence_events=[id_to_event[event_id] for event_id in start_sequence]
    for event in start_sequence_events:
        print(event)"""


    def get_next_filename(base_name, extension):
        i = 0
        while os.path.exists(f"{base_name} {i}.{extension}"):
            i += 1
        return f"{base_name} {i}.{extension}"

    # Generate the next available filename


    for i in range(100):



        try:

            print("\n\n\n GENERATION "+ str(i)+ "\n\n\n")

            random.shuffle(filenames)
            random.shuffle(filenames)
            random.shuffle(filenames)
            random.shuffle(filenames)

            sequence=notes_to_events(filenames[0])

            sequence=events_to_sequence(sequence, event_to_id)

            start_sequence=sequence[:1024]

            output_midi_path = get_next_filename(generated_files_directory, 'mid')

            generated_sequence = generate_sequence(
                model=model,
                event_to_id=event_to_id,
                start_sequence=start_sequence,
                id_to_event=id_to_event,
                output_midi_path=output_midi_path,
                num_generate=10000,      # Number of IDs to generate
                temperature=1.0,        # Adjust for more or less randomness
                sequence_length=1024
            )
        except:
            pass

    """
    generated_sequence = generate_sequence(
        model=model,
        event_to_id=event_to_id,
        start_sequence=start_sequence,
        id_to_event=id_to_event,
        output_midi_path='transformergenerated.mid',
        num_generate=3000,      # Number of IDs to generate
        temperature=1.0,        # Adjust for more or less randomness
        sequence_length=1024
    )
    """
