Hey,

So my model runs fine without any errors but the accuracy is really bad. Like it barely learns anything -- the validation accuracy stays super low and doesn't really improve much across epochs. Also early stopping kicks in almost immediately so it only trains for like 1-2 epochs before stopping. I tried running it a couple times and it's the same thing every time.

I'm using DenseNet121 with transfer learning and the model compiles and trains without crashing, the loss just doesn't go down the way I'd expect it to. Not sure what's going on.

Thanks
