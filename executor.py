from jina import Executor, DocumentArray, requests
import face_recognition
from PIL import Image
import os

def check_human(image_uri):
    """check_human.

    Detects if there's one or more humans in the image. 
    :param image_uri: Path to image file
    """
    image = face_recognition.load_image_file(image_uri)
    face_locations = face_recognition.face_locations(image)
    if face_locations:
        return True
    else:
        return False

class FaceDetector(Executor):
    @requests(on="/search")
    def check_human(self, docs: DocumentArray, **kwargs):
        """check_human.
        Checks for a human face in an image Document, and writes True or False to doc.tags["is_human"]
        :param docs:
        :type docs: DocumentArray
        :param kwargs:
        """
        for doc in docs:
            # First ensure we have a blob
            if hasattr(doc, "uri"):
                doc.convert_uri_to_blob()
            elif hasattr(doc, "buffer"):
                doc.convert_buffer_to_blob()

            assert hasattr(doc, "blob")

            # Dump blob to image file
            image = Image.fromarray(doc.blog)
            image.save("doc_content.jpg")

            doc.tags["is_human"] = check_human("doc_content.jpg")
            os.remove("doc_content.jpg")
