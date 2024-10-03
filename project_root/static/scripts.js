document.addEventListener("DOMContentLoaded", function () {
    const pdfUpload = document.getElementById('pdfUpload');
    const pdfLabel = document.getElementById('pdfLabel');
    const pdfFileName = document.getElementById('pdfFileName');
    
    const kbUpload = document.getElementById('knowledgeBase');
    const kbLabel = document.getElementById('kbLabel');
    const kbFileName = document.getElementById('kbFileName');

    // Event listener for PDF upload
    pdfUpload.addEventListener('change', function () {
        if (pdfUpload.files.length > 0) {
            const fileName = pdfUpload.files[0].name;
            pdfFileName.textContent = `File added: ${fileName}`;
            pdfLabel.textContent = "File selected";
        } else {
            pdfFileName.textContent = '';
            pdfLabel.textContent = "Drag & drop or click to choose files";
        }
    });

    // Event listener for Knowledge Base upload
    kbUpload.addEventListener('change', function () {
        if (kbUpload.files.length > 0) {
            const fileName = kbUpload.files[0].name;
            kbFileName.textContent = `File added: ${fileName}`;
            kbLabel.textContent = "File selected";
        } else {
            kbFileName.textContent = '';
            kbLabel.textContent = "Drag & drop or click to choose files";
        }
    });

    // Form submission and processing
    document.getElementById('uploadForm').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent page reload on form submit

        const formData = new FormData(this);
        const resultDiv = document.getElementById('result');
        const processButton = document.getElementById('processButton');
        const buttonText = document.getElementById('buttonText');
        const loader = document.getElementById('loader');

        // Disable button and show loader
        buttonText.textContent = 'Processing...';
        loader.classList.remove('hidden');
        processButton.disabled = true;

        resultDiv.innerHTML = 'Processing...';

        fetch('/process_pdf', {
            method: 'POST',
            body: formData
        })
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'processed_output.pdf';
            document.body.appendChild(a);
            a.click();
            a.remove();
            
            resultDiv.innerHTML = 'Processing complete. Your file is downloading.';
        })
        .catch(error => {
            console.error('Error:', error);
            resultDiv.innerHTML = 'An error occurred. Please try again.';
        })
        .finally(() => {
            // Re-enable button and hide loader after processing
            buttonText.textContent = 'Process Files';
            loader.classList.add('hidden');
            processButton.disabled = false;
        });
    });
});
