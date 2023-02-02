# %%
import tensorflow as tf
import argparse

# %%
base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def sequence2int(sequence, mapping=base2int):
    return [mapping.get(base, 999) for base in sequence]

def sequence2onehot(sequence, mapping=base2int):
    return tf.one_hot(sequence2int(sequence, mapping), depth=4)

def load_fasta(fasta):
    with open(fasta) as f:
        for line in f:
            assert line[0] == '>'
            header, sequence = line.strip(), f.readline().strip()
            name, *_ = header[1:].split(':')
            yield tf.cast(sequence2onehot(sequence), tf.float32)

def load_dataset(fasta):
    dataset = tf.data.Dataset.from_generator(lambda: load_fasta(fasta), output_types=tf.float32)
    dataset = dataset.batch(128)
    return dataset

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta')
    parser.add_argument('-o', '--output')
    parser.add_argument('-m', '--model')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    model.summary()

    dataset = load_dataset(args.fasta)
    print(dataset.element_spec)

    with open(args.output, 'w') as f_out:
        for batch in dataset:
            pred_batch = model(batch)[:, 0].numpy()
            for score in pred_batch:
                print(score, file=f_out)


# %%
if __name__ == '__main__':
    main()