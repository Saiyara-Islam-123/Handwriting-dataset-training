from PIL import Image
import os

def merge(im1, im2, loc, index):
    images = [Image.open(x) for x in [im1, im2]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(f"{loc}/Combined{index}"+".png")
    #new_im.save(f"{loc}/no_train.png")

#merge(im1= "new_plots/blue-green/fast_lr/0 Acc distance plot 0 1.png", im2="new_plots/scatter_plots/no_train/no_train.png", loc="new_plots/combined/fast")


lines = os.listdir("new_plots/blue-green/fast_lr/sup")
scatter_plots = os.listdir("new_plots/scatter_plots/sup_fast")


for i in range(61, 121):
    a = f"new_plots/blue-green/fast_lr/sup/{i} Acc distance plot 0 1.png"
    b = f"new_plots/scatter_plots/sup_fast/{(i-61)*10} 100 0.005 sup_model.png"

    merge(a, b, loc="new_plots/combined/fast", index=i)


#merge("unsup plots unsup=0.05, sup=0.05, per batch/-1 Acc distance plot 0 1.png","unsup plots unsup=0.05, sup=0.05, per batch/599 100 0.05 unsup_model.png")

