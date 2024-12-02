import os
from PIL import Image

def create_gif_from_pngs(folder_path, gif_path, duration=100):
    """
    Reads PNG files from a folder and creates a GIF.
    
    Parameters:
        folder_path (str): Path to the folder containing PNG images.
        gif_path (str): Path where the GIF will be saved.
        duration (int): Duration of each frame in milliseconds (default is 100).
    """
    # Get all PNG files in the folder and sort them (by name)
    png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    png_files.sort()  # Ensure the frames are in the correct order

    if not png_files:
        print("No PNG files found in the specified folder.")
        return

    # Load images
    images = [Image.open(os.path.join(folder_path, file)) for file in png_files]

    # Save the first image as GIF and append the rest
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],  # Add the rest of the images
        duration=duration,  # Duration per frame in milliseconds
        loop=0  # Infinite loop
    )
    print(f"GIF successfully created at {gif_path}")

# Example usage
if __name__ == "__main__":
    folder = "./anim"  # Replace with the folder path containing PNGs
    output_gif = "anim.gif"  # Replace with your desired GIF file name
    create_gif_from_pngs(folder, output_gif)
