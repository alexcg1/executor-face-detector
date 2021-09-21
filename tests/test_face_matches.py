from jina import Document, DocumentArray, Flow
from jina.types.document.generators import from_files
# from ..executor import FaceDetector
from .. import FaceDetector

docs = DocumentArray(from_files("test_data/**/*.jpg"))

for doc in docs:
    doc.tags["uri"] = doc.uri
    doc.convert_image_uri_to_blob()

flow = (
    Flow()
    .add(uses="jinahub+docker://ImageNormalizer")
    .add(uses=FaceDetector, uses_with={"skip_non_faces": False})
)

with flow:
    output = flow.index(inputs=docs, return_results=True)

def test_matches():
    for doc in output[0].docs:
        manual_label = doc.tags["uri"].split("/")[1]
        assert manual_label == str(doc.tags["is_human"])
