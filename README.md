# PURDUE-2026-SPRING
# CNIT566 Project - Conference Knowledge Synthesis Assistant

A knowledge graph-based reflection assistant that helps users synthesize insights across multiple conference sessions.

---

## ■ Overview

Conference attendees often struggle to connect ideas across multiple sessions.  
While they are exposed to many topics, the knowledge remains fragmented and lacks integration.

This project reframes conference learning as a **knowledge synthesis problem** rather than simple information consumption.

==> Goal: Help users build a "big picture" understanding from scattered conference materials.

---

## ■ Problem

- Attendees join many sessions but:
  - Topics are fragmented
  - Connections are not explicit
- Result:
  - Poor retention
  - Weak conceptual understanding
  - Missed research insights

This is fundamentally a **cognitive + learning problem**, not a logistics problem. :contentReference[oaicite:0]{index=0}

---

## ■ Solution

This system builds a **Knowledge Graph-based reflection assistant** that:

- Extracts key concepts from papers and notes
- Identifies relationships between concepts
- Constructs an integrated knowledge graph
- Helps users:
  - Explore connections
  - Compare ideas
  - Detect blind spots

---

## ■ System Architecture

<img width="1536" height="1024" alt="ChatGPT Image Apr 25, 2026, 03_54_09 PM" src="https://github.com/user-attachments/assets/33fa92bb-7be1-4e22-ad15-1754f2d2371f" />


---

## ■ Tech Stack

### Frontend
- React (graph visualization & interaction)
- Figma / Purdue GenStudio (UI design)

### Backend
- Django (API & orchestration)
- Python (data processing & graph logic)

### LLM Integration
- Llama 3.1 (via GenAI API)
- Used for:
  - Concept candidate extraction
  - Relationship generation

### Knowledge Graph
- In-memory Python graph
- Concept merging across documents

---

## ■ Key Design Principle

> The LLM does NOT build the graph.  
> It only proposes candidates — the system validates everything.

---

## ■ Hallucination Control Strategy

- Limit extraction size
  - ~8–12 concepts per paper
  - ~6–10 relationships
- Only keep grounded concepts (must appear in text)
- Validate relationships with evidence
- Separate concept detection from relationship logic

==> Nothing enters the graph without grounding.

---

## ■ Key Features

- 📊 Knowledge Graph Visualization
- 🔗 Cross-paper Concept Alignment
- ⚖️ Idea Comparison
- 🚨 Blind Spot Detection (missing concepts in notes vs papers)

---

## ■ Trade-offs

- LLM extraction can be noisy
- Concept alignment is approximate
- Prototype focuses on interaction, not full automation

---

## ■ Limitations

- Some concepts/relations may be missing
- Strict grounding may reduce coverage
- Cross-paper concept merging is imperfect

---

## ■ Future Work

- Improve concept alignment across papers
- Enhance relation quality
- Scale to larger multi-document datasets

---

## ■ Demo

https://github.com/user-attachments/assets/88128c70-10bb-488e-b047-0310cd6a7708



---
