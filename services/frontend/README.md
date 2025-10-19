# Frontend - NLP Document Library

Modern web interface for the NLP Document Library built with Next.js, React, and TypeScript.

## Tech Stack

- **Next.js 15** - React framework with app router
- **React 19** - UI library
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - High-quality UI components
- **Recharts** - Data visualization
- **Lucide React** - Icon library

## Quick Start

### Prerequisites

- Node.js 18+
- pnpm (or npm/yarn)
- Backend API running on port 8000

### Installation

```bash
# Install dependencies
pnpm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
pnpm dev
```

### Environment Variables

Create `.env.local`:

```env
# API endpoint
NEXT_PUBLIC_API_URL=http://localhost:8000

# Optional: Analytics, etc.
```

## Development

### Available Scripts

```bash
pnpm dev          # Start development server (port 3000)
pnpm build        # Build for production
pnpm start        # Start production server
pnpm lint         # Run Biome linter
pnpm format       # Format code with Biome
```

### Project Structure

```
src/
├── app/                    # Next.js app router pages
│   ├── documents/         # Document management
│   │   ├── [id]/         # Document detail page
│   │   └── page.tsx      # Documents list page
│   ├── statistics/        # Statistics dashboard
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home (redirects to /documents)
├── components/
│   ├── documents/         # Document-related components
│   │   ├── DocumentsView.tsx    # Main documents container
│   │   ├── DocumentsTable.tsx   # Documents list table
│   │   └── UploadCard.tsx       # File upload component
│   ├── statistics/        # Statistics components
│   ├── topbar.tsx         # Navigation bar
│   └── ui/                # Reusable UI components (shadcn/ui)
└── lib/
    ├── api.ts             # API client and types
    └── utils.ts           # Utility functions
```

## Features

### Document Management

**Upload Documents**
- Drag-and-drop interface
- Multi-file upload support
- Real-time upload status
- Supported formats: TXT, PDF, DOCX, RTF

**Document List**
- View all uploaded documents
- Real-time status updates
- Sort by upload date
- Filter and search (coming soon)

**Document Details**
- View full NLP analysis
- Sentiment visualization
- Vocabulary statistics
- Interactive charts

### Statistics Dashboard

- Aggregate statistics across documents
- Sentiment trends over time
- Vocabulary distributions
- Readability metrics

### UI Components

Built with shadcn/ui components:
- `Button` - Interactive buttons
- `Card` - Content containers
- `Table` - Data tables
- `Badge` - Status indicators
- `Progress` - Loading states
- `Dialog` - Modals
- `Tooltip` - Contextual help

## API Integration

### API Client

The `api.ts` module provides a typed client for backend communication:

```typescript
import { apiClient } from '@/lib/api';

// List documents
const { documents } = await apiClient.getUserDocuments('user_id');

// Upload document
const result = await apiClient.uploadDocument('user_id', file);

// Get document details
const doc = await apiClient.getDocument('user_id', 'doc_id');
```

### Type Definitions

```typescript
interface Document {
  _id: string;
  filename: string;
  uploaded_at: string;
  status: 'processing' | 'completed' | 'failed';
  error?: string;
  stats?: DocumentStats;
}

interface DocumentStats {
  vocab_size: number;
  token_count: number;
  type_token_ratio: number;
  doc_sentiment: Record<string, number>;
  sentiment_method: string;
}
```

## Styling

### Tailwind CSS

Utility-first CSS with custom configuration:

```css
/* Custom colors */
--primary: violet-600
--background: gray-50
--foreground: gray-900

/* Typography */
font-family: Geist Sans, system-ui
```

### Component Styling

Example:
```tsx
<div className="rounded-xl border border-black/10 bg-white shadow-sm">
  <div className="p-4">
    Content
  </div>
</div>
```

## Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Import project in Vercel
3. Configure environment variables:
   - `NEXT_PUBLIC_API_URL`: Your production API URL
4. Deploy

```bash
# Or use Vercel CLI
pnpm install -g vercel
vercel
```

### Other Platforms

**Netlify**
```bash
# Build command
pnpm build

# Publish directory
.next
```

**Docker**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install
COPY . .
RUN pnpm build
CMD ["pnpm", "start"]
```

## Configuration Files

### `next.config.ts`

Next.js configuration with Turbopack:

```typescript
const nextConfig = {
  // Configuration options
};
export default nextConfig;
```

### `tsconfig.json`

TypeScript configuration with path aliases:

```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

### `biome.json`

Code formatting and linting:

```json
{
  "formatter": {
    "indentStyle": "space",
    "lineWidth": 100
  }
}
```

## Testing

```bash
# Unit tests (future)
pnpm test

# E2E tests (future)
pnpm test:e2e

# Type checking
pnpm tsc --noEmit
```

## Performance

### Optimization Features

- **Server Components**: Default server-side rendering
- **Image Optimization**: Next.js automatic image optimization
- **Code Splitting**: Automatic route-based splitting
- **Font Optimization**: Geist font with `next/font`

### Bundle Size

```bash
# Analyze bundle
pnpm build
# Check .next/analyze/

# Key metrics:
# - First Load JS: ~90KB (target: <100KB)
# - Total Size: ~500KB
```

## Troubleshooting

### Common Issues

**API Connection Failed**
- Verify backend is running on correct port
- Check `.env.local` has correct API URL
- Ensure CORS is configured in backend

**Build Errors**
```bash
# Clear cache and rebuild
rm -rf .next
pnpm install
pnpm build
```

**Type Errors**
```bash
# Regenerate types
pnpm tsc --noEmit
```

### Development Tips

1. **Hot Reload**: Changes auto-refresh (Turbopack)
2. **Error Overlay**: Detailed error messages in dev
3. **Network Tab**: Monitor API calls in browser DevTools
4. **React DevTools**: Install browser extension

## Contributing

1. Follow existing code style (enforced by Biome)
2. Add types for new components
3. Test with both light/dark themes
4. Ensure mobile responsiveness
5. Document complex logic

## Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [shadcn/ui](https://ui.shadcn.com)
- [TypeScript Handbook](https://www.typescriptlang.org/docs)

---

**Version**: 0.1.0
**License**: [Add license]
**Maintained By**: NLP Digital Humanities Team
