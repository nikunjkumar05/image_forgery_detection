document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loadingState = document.getElementById('loading-state');
    const resultsContainer = document.getElementById('results-container');
    const resultsGrid = document.getElementById('results-grid');
    const btnReset = document.getElementById('btn-reset');

    // Handle Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files);
        }
    });

    // Handle Click to Upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFiles(e.target.files);
        }
    });

    // Reset UI
    btnReset.addEventListener('click', () => {
        resultsContainer.classList.add('hidden');
        resultsGrid.innerHTML = ''; // clear old results
        dropZone.classList.remove('hidden');
        fileInput.value = '';
    });

    async function handleFiles(fileList) {
        if (fileList.length > 10) {
            alert('You can only upload a maximum of 10 images at once.');
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < fileList.length; i++) {
            if (!fileList[i].type.startsWith('image/')) {
                alert('Please upload only image files (JPG, PNG, WEBP).');
                return;
            }
            formData.append('files', fileList[i]);
        }

        // Show loading state
        dropZone.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Prediction failed');
            }

            const data = await response.json(); // Array of results
            
            // Hide loading, show results
            loadingState.classList.add('hidden');
            resultsContainer.classList.remove('hidden');
            resultsGrid.innerHTML = '';

            // Generate DOM elements for each result
            data.forEach((result, idx) => {
                const card = document.createElement('div');
                card.className = 'result-card';

                const statusClass = result.is_authentic ? 'status-authentic' : 'status-forged';
                const verdictText = result.is_authentic ? 'Authentic' : 'Forged / Manipulated';

                card.innerHTML = `
                    <div class="status-banner ${statusClass}">
                        <h2>${verdictText}</h2>
                    </div>
                    <div class="image-comparison">
                        <div class="image-box">
                            <h3>Original Image</h3>
                            <img src="${result.original_image}" alt="Original">
                        </div>
                        <div class="image-box">
                            <h3>MQ-ELA Frequency Map</h3>
                            <img src="${result.ela_image}" alt="MQ-ELA Map">
                        </div>
                    </div>
                `;
                resultsGrid.appendChild(card);
            });

        } catch (error) {
            alert(`Error: ${error.message}`);
            loadingState.classList.add('hidden');
            dropZone.classList.remove('hidden');
        }
    }
});
