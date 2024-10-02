# Pdf-Form-Filler
used bert-base-uncased model from Hugging Face to make a PDF Form Filler

To add a feature that changes the color of the upload button when it is clicked, you can use JavaScript to add a CSS class to the button when it is clicked. This class will change the button's color.

### Updated `styles.css`

First, define the new CSS class for the button's clicked state:

```css:project/static/css/styles.css
/* static/css/styles.css */
body {
    font-family: Arial, sans-serif;
    background-color: #f4f4f4;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

.container {
    background: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    width: 400px;
    text-align: center;
}

h1 {
    margin-bottom: 20px;
}

form {
    display: flex;
    flex-direction: column;
}

label {
    margin: 10px 0 5px;
}

input[type="file"] {
    margin-bottom: 20px;
}

button {
    padding: 10px;
    background: #007BFF;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

button:hover {
    background: #0056b3;
}

button.clicked {
    background: #28a745; /* Change to green when clicked */
}

#output {
    margin-top: 20px;
}
```

### Updated `scripts.js`

Next, update the JavaScript to add the `clicked` class to the button when it is clicked:

```javascript:project/static/js/scripts.js
// static/js/scripts.js
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append('pdf', document.getElementById('pdf').files[0]);
    formData.append('knowledge_base', document.getElementById('knowledge_base').files[0]);

    // Change the button color when clicked
    const submitButton = event.target.querySelector('button[type="submit"]');
    submitButton.classList.add('clicked');

    fetch('/process_pdf', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'filled_form.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);

        // Reset the button color after download
        submitButton.classList.remove('clicked');
    })
    .catch(error => {
        console.error('Error:', error);
        // Reset the button color in case of error
        submitButton.classList.remove('clicked');
    });
});
```

### Explanation:

1. **CSS**:
   - Added a new CSS class `.clicked` that changes the button's background color to green (`#28a745`).

2. **JavaScript**:
   - Added code to add the `clicked` class to the submit button when the form is submitted.
   - Removed the `clicked` class after the file is downloaded or if an error occurs.

With these changes, the upload button will change color when clicked and revert back to its original color after the file is downloaded or if an error occurs.
