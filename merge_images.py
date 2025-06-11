from PIL import Image

def merge(im1, im2):
    images = [Image.open(x) for x in [im1, im2]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save("1.png")

#for i in range(0, 600, 25):
merge("unsup plots unsup=0.05, sup=0.005, every 25 batches/-25 Acc distance plot 0 1.png", "unsup plots unsup=0.05, sup=0.005, every 25 batches/599 100 0.05 unsup_model.png")