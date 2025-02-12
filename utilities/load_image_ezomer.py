import ezomero
import tifffile
conn = ezomero.connect(
    user="maarten",
    password="",
    host="localhost",
    port=4064,
    group="users",
    secure=True
)
image_id = 2901
img, pixels = ezomero.get_image(conn,image_id)
#dimension order of the numpy array: TZYXC
# Get number of channels
n_channels = pixels.shape[4]
print(f"Number of channels: {n_channels}")
print(f"Pixels array shape: {pixels.shape}")

# Save each channel separately
for c in range(n_channels):
    image_shape = pixels.shape
    channel_img = pixels[0,0,:,:,c]
    tifffile.imwrite(f'{image_id}_c{c+1}.tiff', channel_img)