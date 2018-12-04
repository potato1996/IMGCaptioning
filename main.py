import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from Decoder import Decoder
from Encoder import Encoder
from bleu import evaluate
from dataloader import *
from vocabloader import vocab_loader

# hyper-parameters
batch_size = 64
embedding_size = 512
vocal_size = 9957
learning_rate = 0.001
num_epochs = 50
# argument
log_step = 100

check_point = False
saving_model_path = "/scratch/dd2645/cv-project/models"
encoder_model_path = saving_model_path + "/encoder-1.ckpt"
decoder_model_path = saving_model_path+ "/decoder-1.ckpt"

train_image_file = "/scratch/dd2645/mscoco/train2014"
train_captions_json = "/scratch/dd2645/mscoco/annotations/captions_train2014.json"
val_image_file = "/scratch/dd2645/mscoco/val2017"
val_captions_json = "/scratch/dd2645/mscoco/annotations/captions_val2017.json"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Transform image size to 224
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# Build the models
encoder = Encoder(embed_size=embedding_size).to(device)
decoder = Decoder(vocal_size).to(device)

# load model from a check point
if check_point:
    encoder.load_state_dict(torch.load(encoder_model_path))
    decoder.load_state_dict(torch.load(decoder_model_path))

# Load all vocabulary in the data set
vocab = vocab_loader("vocab.txt")
# Load training data
train_loader = get_train_loader(vocab, train_image_file, train_captions_json, transform, batch_size, True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, list(encoder.parameters()) + list(decoder.parameters())), lr=learning_rate)


def train(epoch):
    encoder.train(True)
    decoder.train(True)
    # Train the models
    total_step = len(train_loader)
    for i, (images, captions, lengths) in enumerate(train_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)

        lengths = torch.tensor(lengths).to(device)
        # Exclude the "<start>" from loss function
        captions = captions[:, 1:]
        lengths_1 = lengths - 1

        targets = pack_padded_sequence(captions, lengths_1, batch_first=True).data

        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        outputs = outputs[:, 1:, :]
        outputs = pack_padded_sequence(outputs, lengths_1, batch_first=True).data
       
        loss = criterion(outputs, targets)
        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        if i % log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

    # Save the model checkpoints
    torch.save(decoder.state_dict(), os.path.join(
        saving_model_path, 'decoder-{}.ckpt'.format(epoch + 1)))
    torch.save(encoder.state_dict(), os.path.join(
        saving_model_path, 'encoder-{}.ckpt'.format(epoch + 1)))


def validation():
    encoder.eval()
    decoder.eval()
    output_captions = dict()  # Map(ID -> List(sentences))
    ref_captions = dict()  # Map(ID -> List(sentences))

    # Load validation data
    val_loader = get_val_loader(val_image_file, val_captions_json, transform)

    # Iterate over validation data set
    with torch.no_grad():
        for i, (image, captions) in enumerate(val_loader):
            image = image.to(device)
            # captions = cations.to(device)??
            feature = encoder(image)
            # where is the soft-max?
            output_caption = decoder.sample(feature)

            # exclude <pad> <start> <end>
            output_without_nonstring = []
            for idx in output_caption:
                if idx == 2:
                    break
                elif idx <= 3:
                    continue
                else:
                    output_without_nonstring.append(vocab.vec2word(idx))
            output_captions[i] = [" ".join(output_without_nonstring)]
            ref_captions[i] = captions
            #for caption in captions:
            #    caption_tmp = []
            #    for idx_of_word in caption.split():
            #        caption.append(vocab.vec2word(idx_of_word))
            #    caption = " ".join(caption)
            #    ref_captions[i].append(caption)

    bleu_score = evaluate(ref_captions, output_captions)
    print(bleu_score)


def main():
    for epoch in range(num_epochs):
        train(epoch)
        validation()


if __name__ == "__main__":
    main()
