# Quick Start Guide

Get the NLP Document Library running in 4 simple steps.

## Prerequisites

- **Python 3.11+** (required for scientific computing libraries)
- **Node.js 18+** and pnpm
- **MongoDB Atlas account** (free tier works great)

## Setup Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Natural-Language-Processing-for-Digital-Humanities
```

### 2. Run Setup

This installs everything: Python virtual environment, Node.js packages, NLTK data, spaCy models, and more.

```bash
pnpm setup
```

The setup script will:
- ✅ Check system requirements (Python 3.11, Node.js, pnpm)
- 🐍 Create Python virtual environment and install dependencies
- 📦 Download NLTK data packages and spaCy models
- 🌐 Install frontend dependencies with pnpm
- ⚙️ Set up environment files

### 3. Configure MongoDB

Edit the `.env` file with your MongoDB Atlas connection string:

```bash
# Copy the template
cp env.example .env

# Edit with your MongoDB Atlas connection
nano .env  # or use your preferred editor
```

Add your MongoDB Atlas connection string:
```env
MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=nlp_library
```

**Need a MongoDB Atlas account?**
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Create a free account
3. Create a cluster (M0 Sandbox is free)
4. Create a database user
5. Get your connection string

### 4. Start Development

```bash
pnpm dev
```

This starts both the backend API (port 8000) and frontend (port 3000) with live reloading.

## Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## What's Next?

1. **Upload a document** - Try uploading a text file to test the NLP analysis
2. **Explore the API** - Visit http://localhost:8000/docs for interactive API documentation
3. **Read the docs** - Check out [SETUP.md](./SETUP.md) for detailed configuration options

## Troubleshooting

**Setup failed?**
```bash
# Check system requirements
python3.11 --version  # Should be 3.11+
node --version         # Should be 18+
pnpm --version         # Should be 10+

# Re-run setup
pnpm setup
```

**Services won't start?**
```bash
# Check if ports are in use
lsof -i :8000  # Backend
lsof -i :3000  # Frontend

# Stop any existing services
./scripts/stop-dev.sh

# Start fresh
pnpm dev
```

**MongoDB connection issues?**
- Verify your connection string in `.env`
- Check MongoDB Atlas network access (allow 0.0.0.0/0 for development)
- Ensure database user has read/write permissions

## Need Help?

- **Detailed Setup**: See [SETUP.md](./SETUP.md) for comprehensive installation instructions
- **API Documentation**: Check [services/api/API_DOCS.md](./services/api/API_DOCS.md)
- **Team Resources**: See [TEAM_CONTRACT.md](./TEAM_CONTRACT.md)

---

**That's it!** You should now have a fully functional NLP Document Library running locally.
