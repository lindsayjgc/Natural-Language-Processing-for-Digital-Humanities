# NLP Document Library for Digital Humanities

A full-stack application for analyzing literary and historical texts using Natural Language Processing. Upload documents, perform automated NLP analysis including sentiment analysis, vocabulary statistics, and more through an intuitive web interface.

## 🚀 Quick Start

```bash
# Clone and navigate
git clone <repository-url>
cd Natural-Language-Processing-for-Digital-Humanities

# Setup everything (Python venv, Node packages, NLTK data, etc.)
pnpm setup

# Configure environment (copy and edit .env file)
cp env.example .env
# Edit .env with your MongoDB connection string and other settings

# Start all services
pnpm dev

# Open your browser
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

**That's it!** Upload a document and explore the NLP analysis features.

> **New to the project?** See [QUICKSTART.md](./QUICKSTART.md) for a detailed 4-step setup guide.

### 📝 Common Commands

```bash
# Setup & Development
pnpm setup            # Complete project setup (first time only)
pnpm dev              # Start both backend and frontend
pnpm backend          # Start backend only
pnpm frontend         # Start frontend only

# Testing
pnpm test             # Run all tests
pnpm test:backend     # Run backend tests only
pnpm test:frontend    # Run frontend tests only
pnpm test:watch       # Run frontend tests in watch mode

# Building
pnpm build            # Build frontend for production

# Code Quality
pnpm lint             # Run linter
pnpm format           # Format code

# Maintenance
pnpm clean            # Clean dependencies and artifacts
./scripts/stop-dev.sh         # Stop all services
```

### Alternative: Shell Scripts

If you prefer to run setup manually:
```bash
./scripts/install.sh          # Install dependencies
./scripts/start-dev.sh        # Start all services
./scripts/stop-dev.sh         # Stop all services
```

---

## 🎯 What You Get

### 📊 NLP Analysis Features
- **Sentiment Analysis** - Positive/negative/neutral sentiment detection
- **Vocabulary Statistics** - Word counts, type-token ratios, readability metrics
- **N-gram Analysis** - Most common words, phrases, and patterns
- **Emotional Tone Detection** - Identify emotional content in text
- **Part-of-Speech Tagging** - Analyze grammatical structures

### 🔐 User Authentication
- **Secure Login/Registration** - JWT-based authentication system
- **Remember Me Feature** - Stay logged in for 30 days with extended tokens
- **Modal-based Forms** - Clean, non-disruptive authentication experience
- **Password Security** - Bcrypt password hashing
- **Session Management** - Automatic token refresh and validation

### 📄 Document Support
- **Multiple Formats** - TXT, PDF, DOCX, RTF, legacy DOC files
- **Smart Processing** - Automatic text cleaning and normalization
- **Project Gutenberg** - Specialized handling for public domain texts
- **Batch Processing** - Upload and analyze multiple documents

### 🌐 Modern Web Interface
- **Drag & Drop Upload** - Intuitive document uploading
- **Real-time Processing** - Live status updates during analysis
- **Interactive Visualizations** - Charts and graphs for NLP statistics
- **Responsive Design** - Works on desktop and mobile devices
- **User Authentication** - Secure login/registration with "Remember Me" functionality
- **Modal-based Auth** - Clean, non-disruptive authentication experience

## 🎯 Project Overview

This system allows researchers and students in digital humanities to:
- Upload and manage textual documents (TXT, PDF, DOCX, RTF)
- Automatically process texts with state-of-the-art NLP techniques
- View detailed sentiment analysis and vocabulary statistics
- Analyze patterns across multiple documents
- Store and retrieve analysis results via cloud database

## 📚 Common Use Cases

### For Researchers
- **Literary Analysis** - Analyze sentiment and vocabulary patterns in novels, poetry, plays
- **Historical Document Processing** - Process and analyze historical texts, letters, manuscripts
- **Comparative Studies** - Compare writing styles across different authors or time periods
- **Digital Humanities Projects** - Support large-scale text analysis research

### For Students
- **Text Analysis Learning** - Hands-on experience with NLP techniques
- **Document Processing** - Learn to work with different text formats
- **Research Projects** - Analyze texts for academic assignments
- **Technology Skills** - Gain experience with modern web development and NLP

### For Educators
- **Curriculum Development** - Create interactive text analysis exercises
- **Student Projects** - Provide tools for digital humanities coursework
- **Research Support** - Enable students to perform sophisticated text analysis
- **Technology Integration** - Bridge traditional humanities with modern technology

**Team Documentation**: [Team Contract](./TEAM_CONTRACT.md) | [Meeting Notes](https://docs.google.com/document/d/1zTYazFBrUcNKSXYYG3Sil48KUg2LX25lfew50AlFTLI/edit) | [Project Proposal](https://docs.google.com/document/d/1VOufFqSP2i1Cwe3wolJzSFsuPW4WJDnXQBxCUmPc7Qw/edit)

## 🏗️ Architecture

### Technology Stack

**Backend**
- **FastAPI** - Modern Python web framework
- **MongoDB Atlas** - Cloud NoSQL database
- **Motor** - Async MongoDB driver

**NLP Pipeline**
- **spaCy** - Industrial-strength NLP
- **NLTK** - Natural Language Toolkit
- **Transformers** - Hugging Face transformer models for sentiment analysis

**ETL Pipeline**
- Custom text processing and normalization
- Multi-format document readers (PDF, DOCX, RTF, TXT)
- Project Gutenberg boilerplate removal

**Frontend**
- **Next.js 15** - React framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS
- **shadcn/ui** - Component library

### System Components

```
├── services/
│   ├── api/          # FastAPI backend server
│   ├── nlp/          # NLP processing pipeline
│   ├── etl/          # Text ingestion and normalization
│   ├── frontend/     # Next.js web application
│   └── shared/       # Shared utilities
├── data/             # Sample text corpus
│   ├── literature/   # Literary texts
│   └── history/      # Historical documents
└── tests/            # Test suites
```


## 📚 Documentation

- **[Setup Guide](./SETUP.md)** - Detailed installation instructions
- **[API Documentation](./services/api/API_DOCS.md)** - Complete API reference
- **[ETL Pipeline](./services/etl/README.md)** - Text processing documentation
- **[NLP Pipeline](./services/nlp/README.md)** - NLP analysis documentation

## 🧪 Testing

```bash
# Run all tests
./venv311/bin/python -m pytest tests/ -v

# Run specific test suites
./venv311/bin/python -m pytest tests/unit/ -v         # Unit tests
./venv311/bin/python -m pytest tests/integration/ -v  # Integration tests
```

## 🔄 Development Workflow

1. **Start development environment:**
   ```bash
   ./scripts/start-dev.sh
   ```

2. **Make changes to code:**
   - Backend changes in `services/api/` auto-reload
   - Frontend changes in `services/frontend/src/` hot-reload
   - NLP pipeline changes in `services/nlp/` require backend restart

3. **Test your changes:**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000/docs
   - Upload a document to test the full pipeline

4. **Stop when done:**
   ```bash
   ./scripts/stop-dev.sh
   ```

### File Watching & Hot Reload

- **Backend**: Uvicorn auto-reloads on Python file changes
- **Frontend**: Next.js hot module replacement for instant updates
- **Logs**: Real-time colored output shows both services
- **Database**: MongoDB Atlas persists data between restarts

## 🛠️ Development

### Project Structure

```
services/
├── api/
│   ├── api.py          # Main FastAPI application
│   ├── database.py     # MongoDB operations
│   └── test_api.py     # Test version without MongoDB
├── nlp/
│   ├── analyze_texts.py   # Main analysis pipeline
│   ├── preprocessing.py   # Text preprocessing
│   ├── features.py        # Feature extraction
│   └── sentiment.py       # Sentiment analysis
├── etl/
│   ├── readers.py         # Multi-format document readers
│   ├── normalizers.py     # Text normalization
│   └── ingest_texts.py    # ETL CLI
└── frontend/
    └── src/
        ├── app/              # Next.js pages
        ├── components/       # React components
        └── lib/             # Utilities and API client
```

### Key Features

**NLP Analysis**
- Sentiment analysis using transformer models (with VADER fallback)
- Vocabulary statistics (vocab size, token count, type-token ratio)
- Part-of-speech tagging
- N-gram extraction
- Emotional tone detection

**Document Processing**
- Multi-format support: TXT, PDF, DOCX, RTF, legacy DOC
- Automatic text cleaning and normalization
- Project Gutenberg boilerplate removal
- Footnote removal
- Whitespace normalization

**Web Interface**
- Drag-and-drop document upload
- Real-time processing status
- Interactive NLP statistics visualization
- Document management (list, view, analyze)
- User authentication with "Remember Me" functionality
- Modal-based login/registration forms
- Responsive design for mobile/desktop

## 🌐 Deployment

The application is designed for cloud deployment:

- **Backend**: Deploy to Railway, Render, Fly.io, or cloud providers
- **Frontend**: Deploy to Vercel (recommended) or Netlify
- **Database**: MongoDB Atlas (already configured)

See individual service READMEs for deployment-specific instructions.

## 📊 API Endpoints

### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login (supports "Remember Me")
- `GET /auth/me` - Get current user info

### Documents
- `GET /` - Health check
- `GET /documents/{user_id}` - List user documents
- `POST /documents/upload` - Upload and process document
- `GET /documents/{user_id}/{item_id}` - Get document with NLP stats

Full API documentation: [API_DOCS.md](./services/api/API_DOCS.md)

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| **Script permission denied** | `chmod +x start-dev.sh stop-dev.sh install.sh` |
| **Port already in use** | `./scripts/stop-dev.sh --verbose` |
| **Installation failed** | `./scripts/install.sh --yes` (retry with auto-mode) |
| **Services won't start** | Check `.env` file and MongoDB connection |
| **Frontend not loading** | Verify `NEXT_PUBLIC_API_URL` in `.env.local` |

> **Need more help?** See [SETUP.md](./SETUP.md) for detailed troubleshooting guide.

## 📈 Project Status

### ✅ What's Working
- **Complete Installation System** - Automated setup with `./scripts/install.sh`
- **Development Environment** - Easy start/stop with `./scripts/start-dev.sh` and `./scripts/stop-dev.sh`
- **Full NLP Pipeline** - Sentiment analysis, vocabulary stats, n-grams, POS tagging
- **Multi-format Support** - TXT, PDF, DOCX, RTF document processing
- **Modern Web Interface** - React/Next.js frontend with real-time updates
- **User Authentication** - Secure login/registration with JWT tokens
- **Remember Me Feature** - Extended session tokens (30 days) for convenience
- **Cloud Database** - MongoDB Atlas integration for data persistence
- **API Documentation** - Interactive API docs at `/docs`

### 🚧 In Development
- **Batch Processing** - Upload and analyze multiple documents simultaneously
- **Advanced Visualizations** - Enhanced charts and graphs for NLP statistics
- **Document Sharing** - Multi-user document collaboration features
- **Export Features** - Download analysis results in various formats

### 🎯 Roadmap
- **Collaborative Features** - Team workspaces and shared document libraries
- **Advanced NLP Models** - Integration with latest transformer models
- **Mobile App** - Native mobile application for document analysis
- **API Rate Limiting** - Production-ready API with proper rate limiting

## 🤝 Contributing

See [Team Contract](./TEAM_CONTRACT.md) for team collaboration guidelines.

### Development Setup
```bash
# Clone and setup
git clone <repository-url>
cd Natural-Language-Processing-for-Digital-Humanities
./scripts/install.sh

# Start developing
./scripts/start-dev.sh

# Run tests
./venv311/bin/python -m pytest tests/ -v
```

## 📝 License

[Add your license here]

## 🙏 Acknowledgements

- Project Gutenberg for public domain texts
- The spaCy, NLTK, and Hugging Face teams for excellent NLP tools
- Contributors to docx2txt, python-docx, and textract libraries

---

**Course**: Digital Humanities NLP Project
**Institution**: [Your Institution]
**Year**: 2025
