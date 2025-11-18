# Zeus Admin Dashboard - Implementation Guide

## Overview

A comprehensive admin dashboard has been successfully implemented for the Zeus Splunk Query LLM system. This dashboard provides complete visibility and control over the system, including analytics, query management, feedback curation, system monitoring, and model training capabilities.

## What Was Implemented

### 1. Database Model Addition

**File: `/Users/mojonito/Projects/Zeus/src/database/models.py`**

Added `TrainingJob` model to track model retraining jobs with the following fields:
- `id` - Primary key
- `status` - Job status (pending/running/completed/failed)
- `config` - Training configuration as JSON
- `start_time` - Job start timestamp
- `end_time` - Job completion timestamp
- `model_output_path` - Path to trained model
- `metrics` - Training metrics as JSON
- `error_message` - Error details if job failed
- `created_by` - Foreign key to User (admin who created the job)
- `created_at` - Creation timestamp

### 2. Backend API Endpoints

**File: `/Users/mojonito/Projects/Zeus/src/inference/server.py`**

#### Admin Authorization
- Created `admin_required()` dependency that checks `user.is_admin = True`

#### Analytics Endpoints
- `GET /api/admin/analytics` - Overall system statistics including:
  - Total queries, users, active users
  - Feedback metrics and approval rate
  - Query activity by time period (today, 7d, 30d)
  - Average queries per user

- `GET /api/admin/queries` - List all queries with filters:
  - Query parameters: `page`, `page_size`, `user_id`, `rating`, `start_date`, `end_date`
  - Returns paginated list with user info and feedback status

- `GET /api/admin/feedback` - List all feedback with filters:
  - Query parameters: `page`, `page_size`, `rating`, `user_id`
  - Returns paginated list with full feedback details

#### Feedback Management Endpoints
- `PUT /api/admin/feedback/{id}` - Update feedback:
  - Edit `corrected_query` and `comment` fields
  - Updates `updated_at` timestamp

- `DELETE /api/admin/feedback/{id}` - Delete poor quality feedback

- `POST /api/admin/feedback/export` - Export feedback as training data:
  - Returns JSONL format file
  - Filter by rating (optional)
  - Includes instruction, input, and output fields
  - Uses corrected query for bad feedback with corrections

#### System Monitoring Endpoints
- `GET /api/admin/system/stats` - Real-time system statistics:
  - CPU usage percentage
  - Memory usage (total, used, percentage)
  - Disk usage (total, used, percentage)
  - System uptime in seconds

- `GET /api/admin/system/metrics` - Application metrics:
  - Total requests count
  - Average latency (placeholder for now)
  - Error rate (placeholder for now)
  - Requests per minute

#### Model Training Endpoints
- `POST /api/admin/training/start` - Create new training job:
  - Accepts configuration JSON
  - Options for using feedback data
  - Creates job record with "pending" status
  - Note: Actual training execution requires background worker implementation

- `GET /api/admin/training/jobs` - List training job history:
  - Paginated list of all training jobs
  - Ordered by creation date (newest first)

- `GET /api/admin/training/jobs/{id}` - Get specific job details:
  - Full job information including status, metrics, and errors

### 3. Dependencies Added

**File: `/Users/mojonito/Projects/Zeus/requirements.txt`**

- Added `psutil>=5.9.0` for system monitoring capabilities

### 4. Frontend Dashboard

**File: `/Users/mojonito/Projects/Zeus/web/admin.html`**

A professional, modern admin dashboard with the following features:

#### Design
- Built with Tailwind CSS for modern, responsive design
- Tab-based navigation for different sections
- Real-time data updates
- Interactive charts using Chart.js

#### Tab Sections

**Analytics Tab:**
- Key metric cards (queries, users, approval rate, avg queries/user)
- Query activity bar chart (today, 7d, 30d)
- Feedback distribution doughnut chart
- Auto-loading on page load

**Queries Tab:**
- Searchable, filterable table of all queries
- Filter by rating (good/bad)
- Pagination support
- Shows user, instruction, generated query, feedback status
- Displays creation timestamp

**Feedback Tab:**
- Complete feedback management interface
- Filter by rating
- Edit modal for updating corrected queries and comments
- Delete functionality with confirmation
- Export button to download training data (JSONL)
- Pagination support

**System Tab:**
- Real-time system resource monitoring
- CPU, Memory, Disk usage with visual progress bars
- System uptime display
- Application metrics (requests, latency, error rate)
- Auto-refresh every 30 seconds when active

**Training Tab:**
- Form to start new training jobs
- Configuration options (use feedback data, minimum rating)
- JSON configuration input for advanced settings
- Training job history table
- Job status indicators (pending/running/completed/failed)
- Error message display for failed jobs

#### Security Features
- Login check on page load
- Admin privilege verification
- Automatic redirect to chat page if not authenticated or not admin
- JWT token-based authentication
- Token stored in localStorage

#### User Experience
- Responsive design works on all screen sizes
- Loading states for all async operations
- Error handling with user-friendly alerts
- Smooth transitions and hover effects
- Color-coded status badges (green/red for good/bad)

## Database Migration Required

After implementing this admin dashboard, you'll need to run a database migration to create the `training_jobs` table:

```bash
# If using Alembic (recommended)
alembic revision --autogenerate -m "Add TrainingJob model"
alembic upgrade head

# Or manually create the table
# The SQL schema will be automatically created by SQLAlchemy on first access
```

## Setting Up an Admin User

To access the admin dashboard, you need a user account with `is_admin = True`. You can create one by:

1. Register a normal user through the API or web interface
2. Manually update the database to set `is_admin = True`:

```sql
UPDATE users SET is_admin = TRUE WHERE username = 'your_username';
```

Or using Python:
```python
from src.database.database import get_db
from src.database.models import User

db = next(get_db())
user = db.query(User).filter(User.username == 'your_username').first()
user.is_admin = True
db.commit()
```

## Accessing the Admin Dashboard

1. Start the server:
```bash
python -m src.inference.server --model-path /path/to/model
```

2. Open your browser to:
```
http://localhost:8000/web/admin.html
```

3. Login with admin credentials
4. The dashboard will verify admin privileges and load

## API Testing

You can test the admin endpoints using curl:

```bash
# Get analytics
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/admin/analytics

# List queries
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/admin/queries?page=1&page_size=50

# Export feedback
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/admin/feedback/export -o feedback.jsonl

# Get system stats
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/admin/system/stats

# Start training job
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" -H "Content-Type: application/json" \
  -d '{"use_feedback_data": true, "min_rating": "good"}' \
  http://localhost:8000/api/admin/training/start
```

## Future Enhancements

The current implementation provides a solid foundation. Consider these enhancements for production:

1. **Background Job Processing:**
   - Integrate Celery or similar for actual model training execution
   - Implement progress tracking for long-running jobs
   - Add job cancellation capability

2. **Advanced Metrics:**
   - Implement proper request tracking middleware
   - Add response time percentiles (p50, p95, p99)
   - Track model inference latency separately

3. **Enhanced Analytics:**
   - User activity heatmaps
   - Query complexity analysis
   - A/B testing results for model improvements
   - Cost tracking (if using paid infrastructure)

4. **Notifications:**
   - Email alerts for failed training jobs
   - Slack/webhook integration for system alerts
   - Dashboard notifications for important events

5. **Audit Logging:**
   - Track all admin actions
   - Export audit logs
   - Compliance reporting

6. **Data Export:**
   - Schedule automated backups
   - Export to different formats (CSV, JSON, Parquet)
   - Direct integration with data warehouses

## Files Modified

1. `/Users/mojonito/Projects/Zeus/src/database/models.py` - Added TrainingJob model
2. `/Users/mojonito/Projects/Zeus/src/inference/server.py` - Added admin endpoints and authorization
3. `/Users/mojonito/Projects/Zeus/requirements.txt` - Added psutil dependency
4. `/Users/mojonito/Projects/Zeus/web/admin.html` - Created admin dashboard frontend

## Testing Checklist

- [ ] Database migration completed successfully
- [ ] Admin user created and verified
- [ ] All API endpoints return expected responses
- [ ] Admin authorization blocks non-admin users
- [ ] Frontend loads without console errors
- [ ] Analytics metrics display correctly
- [ ] Query filtering and pagination work
- [ ] Feedback edit/delete operations succeed
- [ ] System stats update in real-time
- [ ] Training job creation works
- [ ] Export functionality downloads JSONL file
- [ ] Logout clears session properly

## Notes

- The system metrics (latency, error rate) use basic placeholders. For production, implement proper request tracking middleware.
- Training job execution is not implemented - only job record creation. You'll need to implement the actual training logic with a background worker.
- All timestamps are stored in UTC.
- The frontend assumes the API is running on `localhost:8000` - update the `API_BASE` constant if deploying elsewhere.

## Support

For issues or questions about the admin dashboard:
1. Check the browser console for JavaScript errors
2. Check the server logs for API errors
3. Verify authentication tokens are valid
4. Ensure admin privileges are set correctly in the database
