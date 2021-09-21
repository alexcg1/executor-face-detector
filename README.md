# FaceDetector

Note: WIP, not fully baked yet

Detects if a human face is in an image Document and stores `True` or `False` in `doc.tags['is_human']`.

**Note**: It recognizes *human faces in general*. It doesn't recognize *individuals*.

In the words of the late, great [Terry Pratchett](http://www.chrisjoneswriting.com/terry-pratchett-quotes/technology-terry-pratchett-quote):

> Oh, well, if you prefer, I can recognize handwriting,’ said the imp proudly.  ‘I’m quite advanced.’
> Vimes pulled out his notebook and held it up. ‘Like this?’ he said.
> The imp squinted for a moment. ‘Yep,’ it said. ‘That’s handwriting, sure enough. Curly bits, spiky bits, all joined together. Yep. Handwriting. I’d recognize it anywhere.'

## Why would you use this?

- You're indexing your personal photo collection and later in the Flow you'll use a facial-recognition encoder so you can search by an individual's face. This can filter out all the non-faces, thus saving compute in the encoding step.
- You're creating a human face search engine (e.g. find your celebrity twin) and you don't want trolls getting cheap laughs from uploading inappropriate pictures and seeing who matches.

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
