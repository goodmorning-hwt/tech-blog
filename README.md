# How to Maintain This Blog

A quick reminder for my future self on how to write, preview, and publish a new post.

**Tech Stack:** Hugo + PaperMod Theme + GitHub Pages
**Source Code:** `master` branch
**Live Website:** Deployed from `gh-pages` branch

-----

### 1\. Writing & Previewing Locally

1.  **Create a New Post:**

    ```bash
    # For an English post
    hugo new en/posts/my-cool-post.md

    # For a Chinese post
    hugo new zh/posts/my-cool-post.md
    ```

2.  **Edit the New File:**

      * Find the new file inside the `content/en/posts/` or `content/zh/posts/` directory.
      * **Important:** Change `draft: true` to `draft: false` in the file's header to make it visible.

3.  **Preview the Site:**

      * Run the local server:
        ```bash
        hugo server
        ```
      * Open your browser to `http://localhost:1313` to see your changes live.

### 2\. Publishing Your Changes

1.  **Commit Your Source Code:**

      * First, save your writing progress to the `main` branch.
      * Press `Ctrl + C` to stop the server.

    <!-- end list -->

    ```bash
    git add .
    git commit -m "Wrote a new post about..."
    git push origin main
    ```

2.  **Deploy to the Live Site:**

      * Run this single command. It will build the site and push it to the `gh-pages` branch automatically.

    <!-- end list -->

    ```bash
    npm run deploy
    ```

That's it. Wait a minute or two for GitHub Pages to update. Happy blogging\!
