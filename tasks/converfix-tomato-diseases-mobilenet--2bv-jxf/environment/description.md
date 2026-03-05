hey everyone, first time posting here so sorry if i mess up the format

im trying to classify tomato diseases using MobileNetV3Small with transfer learning and im getting really bad accuracy. like its training fine, no errors or anything, loss goes down a bit but the validation accuracy is stuck around 30-40% and barely improves after the first few epochs. i expected at least 80%+ since im using a pretrained model on imagenet

i tried increasing epochs but early stopping kicks in because it just plateaus. the model seems to predict the same few classes for everything and doesnt really learn the differences between the diseases

has anyone had a similar issue with mobilenet transfer learning? any help would be appreciated thanks
