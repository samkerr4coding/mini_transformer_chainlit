# Miniature Transformer Sentence Completion with Chainlit UI

## Global Design

![mermaid-diagram-global-design-2025-09-10-133158.png](docs/images/mermaid-diagram-global-design-2025-09-10-133158.png)

## Setup

### With pip

```bash
pip install -r requirements.txt
```

##  Train the model:

![mermaid-diagram-training-2025-09-10-135701.png](docs/images/mermaid-diagram-training-2025-09-10-135701.png)

```python
python train.py
```

## Start the Chainlit UI:
chainlit run app.py

Open the UI at the URL provided, and enter a phrase like the cat sat to see completion.
