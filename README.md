# FaceDetector

Detects if a human face is in an image Document.

**Note**: It recognizes *human faces in general*. It doesn't recognize *individuals*.

In the words of the late, great [Terry Pratchett](http://www.chrisjoneswriting.com/terry-pratchett-quotes/technology-terry-pratchett-quote):

> Oh, well, if you prefer, I can recognize handwriting,’ said the imp proudly.  ‘I’m quite advanced.’
> Vimes pulled out his notebook and held it up. ‘Like this?’ he said.
> The imp squinted for a moment. ‘Yep,’ it said. ‘That’s handwriting, sure enough. Curly bits, spiky bits, all joined together. Yep. Handwriting. I’d recognize it anywhere.'

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://FaceDetector')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://FaceDetector')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
