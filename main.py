import cv2
import numpy as np
import streamlit as st
import imutils
from PIL import Image
from io import BytesIO, BufferedReader
import skimage

# Sidebar image options
image_operation = st.sidebar.selectbox(
    "Image operations",
    ("resize",
     "rotate",
     "flip",
     "blurr",
     "morphological hats",
     "Erosion",
     "Dilation",
     "Canny Edge Detection",
     "Histogram Equalization",
     "Match Histogram")
)


# Image functions

def download_image(image):
    ret, img_encoded = cv2.imencode(".png", image)  # numpy.ndarray
    srt_encoded = img_encoded.tostring()  # bytes
    img_bytes_io = BytesIO(srt_encoded)  # _io.BytesIO
    img_buffered_reader = BufferedReader(img_bytes_io)  # _io.BufferedReader

    return img_buffered_reader


def resize(image):
    # Slider
    value = st.sidebar.slider("Image width (in pixels)", min_value=100, max_value=800, step=100, value=100)

    # to mantain aspect ratio
    r = value / image.shape[1]
    dim = (value, int(image.shape[0] * r))

    # perform the actual resizing of the image
    resized = imutils.resize(image, width=value)

    # Show image
    st.image(resized, channels="BGR", caption="Modified Image")

    # cropped_image converted to PIL image color
    # result = Image.fromarray(resized.astype('uint8'), 'RGB')

    # img = Image.open(result)

    downloaded_image = download_image(image=resized)
    # Download button
    st.download_button(
        label="Download image",
        data=downloaded_image,
        file_name='resized.jpg',
        mime="image/jpg"
    )


def rotate(image):
    # Slider
    value = st.sidebar.slider("Angle", min_value=0, max_value=180, step=15, value=0)

    # perform the actual resizing of the image
    rotated = imutils.rotate_bound(image, value)

    # Show image
    st.image(rotated, channels="BGR", caption="Modified Image")

    downloaded_image = download_image(image=rotated)
    # Download button
    st.download_button(
        label="Download image",
        data=downloaded_image,
        file_name='rotate.jpg',
        mime="image/jpg"
    )


def flip(image):
    # Flip options
    flip_options = st.sidebar.radio(
        "Flip options",
        ('Horizontally', 'Vertically', 'Vertically & Horizontally'))

    # perform action based on selection
    if flip_options == "Horizontally":
        flipped = cv2.flip(image, 1)
    elif flip_options == "Vertically":
        flipped = cv2.flip(image, 0)
    elif flip_options == "Vertically & Horizontally":
        flipped = cv2.flip(image, -1)

    # show image
    st.image(flipped, channels="BGR", caption="Modified Image")

    downloaded_image = download_image(image=flipped)
    # Download button
    st.download_button(
        label="Download image",
        data=downloaded_image,
        file_name='flipped.jpg',
        mime="image/jpg"
    )


def blurring(image):
    # Slider
    value = st.sidebar.slider("Blurr level", min_value=1, max_value=100, step=1, value=1)

    blurred = cv2.blur(image, (value, value))

    # show image
    st.image(blurred, channels="BGR", caption="Modified Image")

    downloaded_image = download_image(image=blurred)
    # Download button
    st.download_button(
        label="Download image",
        data=downloaded_image,
        file_name='blurred.jpg',
        mime="image/jpg"
    )


def morphological_hats(image):
    morph_options = st.sidebar.radio(
        "Options",
        ('TopHat', 'BlackHat'))

    value = st.sidebar.slider("Kernel size (value x value)", min_value=1, max_value=100, step=1, value=1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))

    if morph_options == "TopHat":
        hat_img = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
    elif morph_options == "BlackHat":
        hat_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

    # show image
    st.image(hat_img, caption="Modified Image")

    downloaded_image = download_image(image=hat_img)

    # Download button
    st.download_button(
        label="Download image",
        data=downloaded_image,
        file_name='morph_hat.jpg',
        mime="image/jpg"
    )


def image_erosion(image):
    # Slider
    value = st.sidebar.slider("Erosion level", min_value=1, max_value=100, step=1, value=1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eroded = cv2.erode(gray.copy(), None, iterations=value)

    # show image
    st.image(eroded, caption="Modified Image")

    downloaded_image = download_image(image=eroded)

    # Download button
    st.download_button(
        label="Download image",
        data=downloaded_image,
        file_name='morph_hat.jpg',
        mime="image/jpg"
    )


def image_dilation(image):
    # Slider
    value = st.sidebar.slider("Dilation level", min_value=1, max_value=100, step=1, value=1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray.copy(), None, iterations=value)

    # show image
    st.image(dilated, caption="Modified Image")

    downloaded_image = download_image(image=dilated)

    # Download button
    st.download_button(
        label="Download image",
        data=downloaded_image,
        file_name='morph_hat.jpg',
        mime="image/jpg"
    )


def canny_edge_detection(image):
    # Sliders
    threshold1 = st.sidebar.slider("threshold1", min_value=1, max_value=255, step=1, value=1)
    threshold2 = st.sidebar.slider("threshold2", min_value=1, max_value=255, step=1, value=1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, threshold1, threshold2)
    edges[np.where(edges != 0)] = 100
    edges[np.where(edges == 0)] = 255
    # show image
    st.image(edges, caption="Modified Image")

    downloaded_image = download_image(image=edges)

    # Download button
    st.download_button(
        label="Download image",
        data=downloaded_image,
        file_name='morph_hat.jpg',
        mime="image/jpg"
    )


def histogram_equalization(image):
    equalized = ""
    # Equalization options
    equalization_options = st.sidebar.radio(
        "Equalization options",
        ('Simple', 'Adaptive'))

    # image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if equalization_options == 'Simple':
        equalized = cv2.equalizeHist(gray)
        # show image
        st.image(gray, caption="Gray Image")
        st.image(equalized, caption="Modified Image")
        downloaded_image = download_image(image=equalized)
    elif equalization_options == "Adaptive":
        # Sliders
        clip = st.sidebar.slider("Clip", min_value=1, max_value=4, step=1, value=1)
        tile = st.sidebar.slider("Tile", min_value=1, max_value=100, step=1, value=1)

        clahe = cv2.createCLAHE(clipLimit=clip,
                                tileGridSize=(tile, tile))

        adaptive_equalized = clahe.apply(gray)

        # show image
        st.image(gray, caption="Gray Image")
        st.image(adaptive_equalized, caption="Modified Image")
        downloaded_image = download_image(image=adaptive_equalized)

    # downloaded_image = download_image(image=equalized)

    # Download button
    st.download_button(
        label="Download image",
        data=downloaded_image,
        file_name='simple_equalized.jpg',
        mime="image/jpg"
    )


def match_histogram(src, ref):
    multi = True if src.shape[-1] > 1 else False  # to check how many channels the picture has, if a color image will have 3 channels and will be set it as 'multi', if a gray scale image will not be 'multi'
    matched = skimage.exposure.match_histograms(src, ref, multichannel=multi)

    # show image
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    st.image(ref, caption="Reference Image")
    matched = cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)
    st.image(matched, caption="Matched Image")




# Upload image
uploaded_file = st.file_uploader("Choose a image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image_copy = opencv_image

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR", caption="Original Image")

    if image_operation == "resize":
        resize(opencv_image_copy)
    elif image_operation == "rotate":
        rotate(opencv_image_copy)
    elif image_operation == "flip":
        flip(opencv_image_copy)
    elif image_operation == "blurr":
        blurring(opencv_image_copy)
    elif image_operation == "morphological hats":
        morphological_hats(opencv_image_copy)
    elif image_operation == "Erosion":
        image_erosion(opencv_image_copy)
    elif image_operation == "Dilation":
        image_dilation(opencv_image_copy)
    elif image_operation == "Canny Edge Detection":
        canny_edge_detection(opencv_image_copy)
    elif image_operation == "Histogram Equalization":
        histogram_equalization(opencv_image_copy)
    elif image_operation == "Match Histogram":

        referenced_image = st.file_uploader("Choose a reference image", type=["png", "jpg", "jpeg"])
        if referenced_image is not None:
            file_bytes = np.asarray(bytearray(referenced_image.read()), dtype=np.uint8)
            referenced_image = cv2.imdecode(file_bytes, 1)
            match_histogram(opencv_image_copy,referenced_image)
