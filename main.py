import argparse
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
batch_size = 512
embedding_size = 512
vocal_size = 9957


# argument
log_step = 100

train_image_file = "/scratch/dd2645/mscoco/train2017"
train_captions_json = "/scratch/dd2645/mscoco/annotations/captions_train2017.json"
val_image_file = "/scratch/dd2645/mscoco/val2017"
val_captions_json = "/scratch/dd2645/mscoco/annotations/captions_val2017.json"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epoch, num_epochs, vocab, train_loader, encoder, decoder, optimizer, criterion, saving_model_path):
    
    encoder.train(True)
    decoder.train(True)
    
    # Train the models
    total_step = len(train_loader)
    for i, (images, captions, lengths) in enumerate(train_loader):
        
        # Copy mini-batch data to GPU
        images = images.to(device)
        captions = captions.to(device)
        lengths = torch.tensor(lengths).to(device)
        
        # Exclude the "<start>" from loss function
        lengths_1 = lengths - 1

        encoder.zero_grad()
        decoder.zero_grad()

        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        outputs = outputs[:, 1:, :]
        outputs = pack_padded_sequence(outputs, lengths_1, batch_first=True).data
       

        targets = captions[:, 1:]
        targets = pack_padded_sequence(targets, lengths_1, batch_first=True).data
        
        loss = criterion(outputs, targets)
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


def validation(vocab, val_loader, encoder, decoder):
    encoder.eval()
    decoder.eval()
    output_captions = dict()  # Map(ID -> List(sentences))
    ref_captions = dict()  # Map(ID -> List(sentences))

    # Iterate over validation data set
    with torch.no_grad():
        for i, (image, captions) in enumerate(val_loader):
            image = image.to(device)
            # captions = cations.to(device)??
            feature = encoder(image)
            output_caption = decoder.sample(feature)
            
            if i == 0:
                print(output_caption)
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
            ref_captions[i] = [ref_caption[0] for ref_caption in captions]

    bleu_score = evaluate(ref_captions, output_captions)
    print(bleu_score)


def main(args):
    print(device)

    # Path to save the trained models
    saving_model_path = args.saving_model_path

    # If path is not empty, set check_out = True
    check_point = True if args.encoder_model_path and args.decoder_model_path else False
    
    # Load all vocabulary in the data set
    vocab = vocab_loader("vocab.txt")   

    # Build the models
    encoder = Encoder(base_model=args.cnn_model, embed_size=embedding_size, init=not check_point, train_cnn=args.train_cnn).to(device)
    decoder = Decoder(vocal_size, input_size=embedding_size).to(device)

    # Transform image size to 224 or 299
    size_of_image = 299 if args.cnn_model == "inception" else 224
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size_of_image),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
    
    # Load training data
    train_loader = get_train_loader(vocab, train_image_file, train_captions_json, transform, batch_size, True)

    # Load validation data
    val_loader = get_val_loader(val_image_file, val_captions_json, transform)

    # load model from a check point
    if check_point:
        encoder.load_state_dict(torch.load(args.encoder_model_path))
        decoder.load_state_dict(torch.load(args.decoder_model_path))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, list(encoder.parameters()) + list(decoder.parameters())), lr=args.learning_rate)
    
    for epoch in range(args.num_epochs):
        train(epoch=epoch, num_epochs=args.num_epochs, vocab=vocab, train_loader=train_loader, encoder=encoder, decoder=decoder, optimizer=optimizer, criterion=criterion, saving_model_path=saving_model_path)
        validation(vocab=vocab, val_loader=val_loader, encoder=encoder, decoder=decoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_model', type=str,
                        default='resnet152',
                        help='model choices passed to Encoder, if it is inception resize to 299')
    # load/save model path
    parser.add_argument('--encoder_model_path', type=str, default='',
                        help='path for encoder model')
    parser.add_argument('--decoder_model_path', type=str, default='',
                        help='path for decoder model')
    parser.add_argument('--saving_model_path', type=str, default='/scratch/dd2645/cv-project/models',
                        help='path to save models')
    # Number of Epochs
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of Epochs')
    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    # Tune CNN or not
    parser.add_argument('--train_cnn', type=bool, default=False,
                        help='argument in Encoder')
    args = parser.parse_args()

    main(args)
