import praw
import datetime
import traceback
import numpy as np

# Reddit API credentials
CLIENT_ID = ""
CLIENT_SECRET = ""
USER_AGENT = ""

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)


def fetch_reddit_user(username):
    try:
        user = reddit.redditor(username)

        # Fetch Profile Information
        user_info = {
            "Username": user.name,
            "ID": user.id,
            "Created (UTC)": datetime.datetime.utcfromtimestamp(user.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            "Karma": user.link_karma + user.comment_karma,
            "Link Karma": user.link_karma,
            "Comment Karma": user.comment_karma,
            "Is Reddit Admin": user.is_employee,
            "Verified Email": user.has_verified_email,
        }

        # Fetch latest 50 posts & comments
        raw_comments = list(user.comments.new(limit=50))
        raw_posts = list(user.submissions.new(limit=50))

        # Process Comments
        comments = []
        comment_times = []
        for comment in raw_comments:
            comments.append({
                "Subreddit": comment.subreddit.display_name,
                "Comment": comment.body,
                "Score": comment.score,
                "Created (UTC)": datetime.datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                "Permalink": f"https://www.reddit.com{comment.permalink}"
            })
            comment_times.append(comment.created_utc)

        # Process Posts
        posts = []
        post_times = []
        for post in raw_posts:
            posts.append({
                "Title": post.title,
                "Subreddit": post.subreddit.display_name,
                "Score": post.score,
                "URL": post.url,
                "Created (UTC)": datetime.datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                "Permalink": f"https://www.reddit.com{post.permalink}"
            })
            post_times.append(post.created_utc)

        # Compute Time Intervals
        post_intervals = np.diff(sorted(post_times)).tolist() if len(post_times) > 1 else []
        comment_intervals = np.diff(sorted(comment_times)).tolist() if len(comment_times) > 1 else []

        # Calculate Metrics
        metrics = {
            "Total Posts": len(raw_posts),
            "Total Comments": len(raw_comments),
            "Average Posts Per Day": len(raw_posts) / ((max(post_times) - min(post_times)) / 86400) if len(post_times) > 1 else 0,
            "Average Comments Per Day": len(raw_comments) / ((max(comment_times) - min(comment_times)) / 86400) if len(comment_times) > 1 else 0,
            "Average Post Time Gap (hrs)": np.mean(post_intervals) / 3600 if len(post_intervals) > 0 else 0,
            "Average Comment Time Gap (hrs)": np.mean(comment_intervals) / 3600 if len(comment_intervals) > 0 else 0,
            "Average Comment Length": np.mean([len(comment.body) for comment in raw_comments]) if raw_comments else 0,
            "Most Active Subreddits (Posts)": list({post.subreddit.display_name for post in raw_posts}),
            "Most Active Subreddits (Comments)": list({comment.subreddit.display_name for comment in raw_comments}),
            "Upvote/Downvote Ratio (Posts)": np.mean([post.upvote_ratio for post in raw_posts]) if raw_posts else 0
        }

        return {
            "User Info": user_info,
            "User Metrics": metrics,
            "Latest Comments": comments,
            "Latest Posts": posts
        }

    except Exception as e:
        #traceback.print_exc()
        return {"Error": str(e)}