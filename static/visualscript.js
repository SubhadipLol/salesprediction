document.querySelectorAll('input[name="year"]').forEach(radio => {
    
    radio.addEventListener('change', function() {
        const yearValue = this.value; // Get the selected year
        console.log('Selected year:', yearValue); // Debugging

        // Send the selected year to the Flask server
        fetch('/visualize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ year: yearValue }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Response data:', data); // Debugging

            if (data.image_1 && data.image_2) {
                // Dynamically update the image sources
                const imageContainer = document.getElementById('image-container');
                imageContainer.innerHTML = `
                    <h2>Price vs Ratings</h2>
                    <img src="${data.image_1}" alt="Price vs Ratings for ${yearValue}" width="600">
                    <h2>Predicted Category Distribution</h2>
                    <img src="${data.image_2}" alt="Predicted Category Distribution for ${yearValue}" width="600">
                `;
            } else {
                alert('No data available for this year.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        });
    });
});
function goBack() {
    window.history.back();
}
