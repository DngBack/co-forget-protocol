# Co-Forgetting Protocol Implementation

This project implements the **Co-Forgetting Protocol** for synchronized memory management in multi-agent systems, as introduced in the paper _"The Co-Forgetting Protocol: Synchronized Memory Management for Multi-Agent Systems" (May 2025)_. It leverages **Pinecone** for vector storage and **CrewAI** for orchestrating agent workflows, targeting improvements in task success rate and memory efficiency on benchmarks such as AgentBench.

## 🚀 Features

- 🔄 Collaborative memory pruning between AI agents
- 📦 Pinecone for embedding storage
- 🧠 CrewAI for agent orchestration and task management
- 🌐 Serper for web-enabled search tools
- ✅ Reported improvements: +12% success rate, -54% memory footprint

## 🛠️ Setup Instructions

### 1. Create a Conda Environment (Python 3.11)

```bash
conda create -n coforgetting python=3.11
conda activate coforgetting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the template and fill in your API keys:

```bash
cp .env.example .env
```

Edit .env:

```bash
PINECONE_API_KEY=your_pinecone_api_key
SERPER_API_KEY=your_serper_api_key
OPENAI_API_KEY=your_openai_api_key
```
