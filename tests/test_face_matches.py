import pytest
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

output_docs = output[0].docs

all_labels = []

for doc in output_docs:
    both_labels = (doc.tags["uri"].split("/")[1], doc.tags["is_human"])
    all_labels.append(both_labels)

@pytest.mark.parametrize("manual_label, model_label", all_labels)
def test_labels_match(manual_label, model_label):
    assert str(manual_label) == str(model_label)
