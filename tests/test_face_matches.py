from jina import Document, DocumentArray, Flow
from jina.types.document.generators import from_files
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

# I want it to test each doc in the list, but don't know how to do that in pytest
def test_matches():
    for doc in output[0].docs:
        manual_label = doc.tags["uri"].split("/")[1] # Label added by hand (i.e. name of subfolder)
        machine_label = str(doc.tags["is_human"]) # Label added by executor. Cast to string so same as manual labels
        assert manual_label == machine_label
