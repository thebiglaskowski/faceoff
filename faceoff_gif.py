import glob
import time
import shutil
import subprocess
import insightface
import warnings
from insightface.app import FaceAnalysis
import cv2
import imageio
import logging
import tempfile
import gradio as gr

assert insightface.__version__ >= '0.7'

# Ignore specific warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Initialize the FaceAnalysis app and swapper
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx')

def load_and_validate_source_image(source_file):
    """
    Load the source image and validate it.

    Args:
        source_file (str): Path to the source image.

    Returns:
        List: List of detected faces in the source image.

    Raises:
        ValueError: If the source image cannot be read or if it doesn't have exactly one face.
    """
    img_source = cv2.imread(source_file)
    if img_source is None:
        logging.error(f"Could not read source image: {source_file}")
        raise ValueError("Could not read source image!")

    faces_source = app.get(img_source)
    if len(faces_source) != 1:
        logging.error(f"Source image should have exactly one face: {source_file}")
        raise ValueError("Source image should have exactly one face!")

    return faces_source

def break_gif_into_frames(destination_file, temp_dir):
    """
    Use ffmpeg to break the GIF into frames and save them as individual images in a temporary folder.

    Args:
        destination_file (str): Path to the destination GIF file.
        temp_dir (str): Path to the temporary directory for storing frames.

    Returns:
        List: List of paths to the extracted frames.

    Raises:
        ValueError: If there is an error breaking the GIF into frames.
    """
    command = f'ffmpeg -i {destination_file} {temp_dir}/%04d.png'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        logging.error(f"Error breaking GIF into frames: {stderr.decode('utf-8')}")
        raise ValueError("Error breaking GIF into frames!")

    frames = glob.glob(f'{temp_dir}/*.png')
    frames.sort()
    return frames

def process_frames(frames, faces_source, temp_dir):
    """
    Process each frame by performing face swapping.

    Args:
        frames (List): List of paths to the frames.
        faces_source (List): List of detected faces in the source image.
        temp_dir (str): Path to the temporary directory.

    Raises:
        ValueError: If there is an error processing the frames.
    """
    for frame_path in frames:
        try:
            img_frame = cv2.imread(frame_path)
            if img_frame is None:
                logging.error(f"Could not read frame: {frame_path}")
                continue

            faces_frame = app.get(img_frame)
            res = img_frame.copy()
            for idx, face in enumerate(faces_frame):
                res = swapper.get(res, face, faces_source[0], paste_back=True)

            # Overwrite the original frame with the face-swapped version
            cv2.imwrite(frame_path, res)
            logging.info(f"Face swapping completed for frame: {frame_path}")
        except Exception as e:
            logging.error(f"Error processing faces: {e}")
            raise ValueError("Error processing faces!") from e

def reassemble_gif(frames, destination_file, temp_dir):
    """
    Reassemble the frames into an animated GIF.

    Args:
        frames (List): List of paths to the frames.
        destination_file (str): Path to the destination GIF file.
        temp_dir (str): Path to the temporary directory.

    Returns:
        str: Path to the output GIF file.

    Raises:
        ValueError: If there is an error reassembling the GIF.
    """
    reader = imageio.get_reader(destination_file)
    num_frames = len(reader)

    default_duration = 0.1  # Set this to the default duration you want to use
    durations = [reader.get_meta_data(i).get('duration', default_duration) for i in range(num_frames)]
    fps = sum(durations) / len(durations) / 1000.0
    ticks_per_frame = round(100 / fps)

    frames_data = []
    for filename in frames:
        frames_data.append(imageio.v2.imread(filename))

    if len(frames_data) != len(durations):
        logging.warning("Mismatch between number of frames and durations detected. Using default duration for all frames.")
        durations = [default_duration] * len(frames_data)

    output_file = 'images/output/gif/output_' + str(int(time.time())) + '.gif'
    imageio.mimsave(output_file, frames_data, duration=durations, loop=0)
    logging.info(f"GIF reassembled for: {destination_file}. Output file: {output_file}")

    return output_file

def process_faces(source_file, destination_file):
    """
    Process the faces by performing face swapping on the destination GIF.

    Args:
        source_file (str): Path to the source image file.
        destination_file (str): Path to the destination GIF file.

    Returns:
        Tuple: (Path to the destination file, Path to the output file) or None if there was an error.
    """
    try:
        faces_source = load_and_validate_source_image(source_file)
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = break_gif_into_frames(destination_file, temp_dir)
            if frames is None:
                return None
            process_frames(frames, faces_source, temp_dir)
            output_file = reassemble_gif(frames, destination_file, temp_dir)
            return destination_file, output_file
    except Exception as e:
        logging.error(f"Error processing faces: {e}")
        return None

def faceoff(source_image, destination_gif):
    """
    Face swapping function to be used in the Gradio interface.

    Args:
        source_image (numpy.ndarray): Source image array.
        destination_gif (gradio.FileWrapper): Destination GIF file.

    Returns:
        Tuple: (Path to the destination file, Path to the output file) or None if there was an error.
    """
    source_file_path = "images/source/source.jpg"  # Assuming it's always a JPEG
    cv2.imwrite(source_file_path, source_image)

    destination_file_path = "images/destination/destination.gif"
    destination_file_path = destination_file_path.replace("\\", "/")
    shutil.move(destination_gif.name, destination_file_path)

    output_file = process_faces(source_file_path, destination_file_path)

    return output_file

# Define the input and output components for the Gradio interface
input_components = [
    gr.inputs.Image(label="Source Image"),
    gr.inputs.File(label="Destination GIF")
]

output_components = [
    gr.outputs.Image(label="Original Destination GIF", type="numpy"),
    gr.outputs.Image(label="Output Image", type="numpy")
]

# Create the Gradio interface
iface = gr.Interface(fn=faceoff, inputs=input_components, outputs=output_components, title="FaceOff Img2GIF", description="Swap faces from a source image into a GIF (not all results will be good)", capture_session=True)
iface.launch(server_port=5001)
