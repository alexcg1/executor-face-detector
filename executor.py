from jina import Executor, DocumentArray, requests
import numpy as np
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
    return False


class FaceDetector(Executor):
    def __init__(self, skip_non_faces, **kwargs):
        super().__init__(**kwargs)
        self.skip_non_faces = skip_non_faces

    @requests
    def check_human(self, docs: DocumentArray, **kwargs):
        """check_human.
        Checks for a human face in an image Document, and writes True or False to doc.tags["is_human"]
        :param docs:
        :type docs: DocumentArray
        :param kwargs:
        """
        docs_to_remove = []
        for idx, doc in enumerate(docs):
            assert hasattr(doc, "blob")

            # Dump blob to image file
            image = Image.fromarray((doc.blob * 255).astype(np.uint8))
            image.save("doc_content.jpg")

            doc.tags["is_human"] = check_human("doc_content.jpg")
            os.remove("doc_content.jpg")

            if self.skip_non_faces == True and doc.tags["is_human"] == False:
                docs_to_remove.append(idx)

        docs_to_remove.reverse()
        for idx in docs_to_remove:
            del docs[idx]
