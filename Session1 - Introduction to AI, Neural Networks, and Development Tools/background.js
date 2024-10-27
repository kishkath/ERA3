// This script can handle background tasks, if needed
chrome.runtime.onInstalled.addListener(() => {
  console.log('QuickSite Builder extension installed');
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'generateWebsite') {
    // This is where you would implement the logic to generate the website
    // For now, we'll just send a mock response
    sendResponse({status: 'success', message: 'Website generated successfully'});
  } else if (request.action === 'publishWebsite') {
    // This is where you would implement the logic to publish the website
    // For now, we'll just send a mock response
    sendResponse({status: 'success', message: 'Website published successfully'});
  } else if (request.action === 'publishToGitHub') {
    publishToGitHub(request.html, request.css, request.userDetails)
      .then(url => sendResponse({success: true, url: url}))
      .catch(error => sendResponse({success: false, error: error.message}));
    return true; // Indicates that the response is asynchronous
  }
});

async function publishToGitHub(html, css, userDetails) {
  // This is a simplified example and would require proper OAuth flow in a real-world scenario
  const token = 'YOUR_GITHUB_PERSONAL_ACCESS_TOKEN';
  const username = 'YOUR_GITHUB_USERNAME';
  const repo = `quicksite-${userDetails.name || userDetails.full_name}`.toLowerCase().replace(/\s+/g, '-');

  // Create or update repository
  await fetch(`https://api.github.com/user/repos`, {
    method: 'POST',
    headers: {
      'Authorization': `token ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      name: repo,
      description: `Website for ${userDetails.name || userDetails.full_name}`,
      auto_init: true,
      private: false
    })
  });

  // Create or update index.html
  await createOrUpdateFile(token, username, repo, 'index.html', html);

  // Create or update styles.css
  await createOrUpdateFile(token, username, repo, 'styles.css', css);

  // Create README.md
  const readme = `# ${userDetails.name || userDetails.full_name}'s Website\n\nThis website was created using QuickSite Builder.`;
  await createOrUpdateFile(token, username, repo, 'README.md', readme);

  // Enable GitHub Pages
  await fetch(`https://api.github.com/repos/${username}/${repo}/pages`, {
    method: 'POST',
    headers: {
      'Authorization': `token ${token}`,
      'Content-Type': 'application/json',
      'Accept': 'application/vnd.github.switcheroo-preview+json'
    },
    body: JSON.stringify({
      source: {
        branch: "main",
        path: "/"
      }
    })
  });

  return `https://${username}.github.io/${repo}`;
}

async function createOrUpdateFile(token, username, repo, path, content) {
  const url = `https://api.github.com/repos/${username}/${repo}/contents/${path}`;
  
  // Check if file exists
  const response = await fetch(url, {
    headers: {
      'Authorization': `token ${token}`,
    }
  });

  let method, body;
  if (response.status === 200) {
    const data = await response.json();
    method = 'PUT';
    body = JSON.stringify({
      message: `Update ${path}`,
      content: btoa(content),
      sha: data.sha
    });
  } else {
    method = 'PUT';
    body = JSON.stringify({
      message: `Create ${path}`,
      content: btoa(content)
    });
  }

  await fetch(url, {
    method: method,
    headers: {
      'Authorization': `token ${token}`,
      'Content-Type': 'application/json',
    },
    body: body
  });
}
