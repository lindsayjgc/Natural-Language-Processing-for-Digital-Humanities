# Authentication System

This document describes the authentication system implemented.

## Overview

The authentication system provides secure user registration, login, and protected access to document management features. It uses JWT (JSON Web Tokens) for stateless authentication and includes both backend API endpoints and frontend components.

## Backend Authentication

### Dependencies Added

The following Python packages were added to `requirements.txt`:

```
python-jose[cryptography]
passlib[bcrypt]
```

### Key Components

#### 1. Authentication Module (`services/api/auth.py`)

- **Password Hashing**: Uses bcrypt for secure password storage
- **JWT Token Management**: Creates and verifies access tokens
- **Authentication Dependencies**: FastAPI dependency for protected routes
- **Pydantic Models**: Type-safe request/response models

#### 2. Database Integration (`services/api/database.py`)

- **User Management Functions**:
  - `create_user()`: Creates new user accounts
  - `get_user_by_email()`: Retrieves user by email for login
  - `get_user_by_id()`: Retrieves user by ID for token validation

#### 3. API Endpoints (`services/api/api.py`)

**Authentication Endpoints:**
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user info

**Protected Document Endpoints:**
- `GET /documents` - Get user's documents (requires authentication)
- `POST /documents/upload` - Upload document (requires authentication)
- `GET /documents/{document_id}` - Get specific document (requires authentication)

### Environment Configuration

Add these variables to your `.env` file:

```bash
SECRET_KEY=your-secret-key-change-in-production-make-it-long-and-random
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Frontend Authentication

### Dependencies Added

The following packages were added to `services/frontend/package.json`:

```json
{
  "js-cookie": "^3.0.5",
  "@types/js-cookie": "^3.0.6"
}
```

### Key Components

#### 1. Authentication Context (`src/contexts/AuthContext.tsx`)

- **Global State Management**: Manages user authentication state
- **Token Management**: Handles JWT token storage and retrieval
- **Authentication Methods**: Login, register, logout, and user refresh

#### 2. Authentication Components

- **`LoginForm`**: User login form with validation
- **`RegisterForm`**: User registration form with validation
- **`AuthDialog`**: Modal dialog for login/register
- **`UserMenu`**: User dropdown menu with logout option
- **`ProtectedRoute`**: Wrapper component for protected pages

#### 3. API Client Updates (`src/lib/api.ts`)

- **Automatic Token Inclusion**: Adds JWT token to all API requests
- **Token Management**: Handles token storage and retrieval
- **Error Handling**: Manages authentication errors and token expiration

### Authentication Flow

1. **User Registration/Login**:
   - User fills out registration or login form
   - Frontend sends credentials to backend
   - Backend validates credentials and returns JWT token
   - Frontend stores token in localStorage

2. **Protected Route Access**:
   - User navigates to protected page
   - `ProtectedRoute` component checks authentication status
   - If not authenticated, shows login dialog
   - If authenticated, renders protected content

3. **API Requests**:
   - All API requests automatically include JWT token in Authorization header
   - Backend validates token on each request
   - If token is invalid/expired, user is redirected to login

## Security Features

### Password Security
- **Bcrypt Hashing**: Passwords are hashed using bcrypt with salt
- **No Plain Text Storage**: Passwords are never stored in plain text

### Token Security
- **JWT Tokens**: Stateless authentication using JSON Web Tokens
- **Token Expiration**: Tokens expire after 30 minutes (configurable)

### API Security
- **Protected Endpoints**: All document operations require authentication
- **User Isolation**: Users can only access their own documents
- **Input Validation**: All inputs validated using Pydantic models

## Usage Examples

### Backend Usage

```python
# Protect an endpoint
@app.get("/protected")
async def protected_route(current_user_id: str = Depends(get_current_user)):
    return {"user_id": current_user_id}

# Create a new user
user_data = UserCreate(
    email="user@example.com",
    password="secure_password",
)
user = await create_user(user_data.email, hashed_password)
```

### Frontend Usage

```tsx
// Use authentication context
const { user, isAuthenticated, login, logout } = useAuth();

// Protect a route
<ProtectedRoute>
  <MyComponent />
</ProtectedRoute>

// Make authenticated API calls
const documents = await apiClient.getUserDocuments();
```

## Database Schema

### Users Collection

```javascript
{
  _id: ObjectId,
  email: String (unique),
  hashed_password: String,
  created_at: DateTime,
}
```
## Deployment Considerations

### Environment Variables

Make sure to set these environment variables in production:

```bash
SECRET_KEY=<strong-random-secret-key>
MONGODB_URI=<your-mongodb-connection-string>
DATABASE_NAME=nlp_library
```

### Security Recommendations

1. **Use HTTPS**: Always use HTTPS in production
2. **Secure Secret Key**: Use a strong, random secret key
3. **Token Expiration**: Consider shorter token expiration times
4. **Rate Limiting**: Implement rate limiting for authentication endpoints
5. **CORS Configuration**: Configure CORS properly for your domain

## Testing

### Backend Tests

```python
# Test user registration
def test_register_user():
    user_data = UserCreate(
        email="test@example.com",
        password="testpassword",
    )
    response = client.post("/auth/register", json=user_data.dict())
    assert response.status_code == 200

# Test protected endpoint
def test_protected_route():
    # First login to get token
    login_response = client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "testpassword"
    })
    token = login_response.json()["access_token"]

    # Use token for protected request
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/documents", headers=headers)
    assert response.status_code == 200
```

### Frontend Tests

```tsx
// Test authentication context
import { render, screen } from '@testing-library/react';
import { AuthProvider } from '@/contexts/AuthContext';

test('shows login form when not authenticated', () => {
  render(
    <AuthProvider>
      <ProtectedRoute>
        <div>Protected Content</div>
      </ProtectedRoute>
    </AuthProvider>
  );

  expect(screen.getByText('Authentication Required')).toBeInTheDocument();
});
```

## Troubleshooting

### Common Issues

1. **Token Expiration**: Users will be automatically logged out when tokens expire
2. **CORS Errors**: Make sure CORS is configured to allow frontend domain
3. **Secret Key**: Make sure SECRET_KEY is set and consistent across deployments
4. **Bcrypt Version**: Must use bcrypt < 4.0.0 for passlib compatibility (see requirements.txt)

### Debug Tips

1. **Check Network Tab**: Look for 401/403 errors in browser dev tools
2. **Verify Token**: Check if JWT token is being sent in requests
3. **Database Logs**: Check MongoDB logs for connection issues
4. **Backend Logs**: Check FastAPI logs for authentication errors

## Future Enhancements

### Potential Improvements

1. **Refresh Tokens**: Implement refresh token mechanism for longer sessions
2. **Password Reset**: Add password reset functionality
3. **Email Verification**: Add email verification for new accounts
4. **Social Login**: Add OAuth providers (Google, GitHub, etc.)
5. **Role-Based Access**: Implement user roles and permissions
6. **Audit Logging**: Add authentication event logging
7. **Two-Factor Authentication**: Add 2FA support for enhanced security

### Performance Optimizations

1. **Token Caching**: Cache user information to reduce database queries
2. **Connection Pooling**: Optimize database connection handling
3. **Rate Limiting**: Implement rate limiting for authentication endpoints
4. **Session Management**: Add session management for better user experience
