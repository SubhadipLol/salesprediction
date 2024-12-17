let upload_check = false;

function checkinput() {
    if (!upload_check) {
        upload_check = true;
        document.getElementById('processFileButton').innerText = "Proceed to next page"; // Change text
    }else if(upload_check){
       window.location.href = '/second';
    }
}

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("processFileButton").addEventListener("click", function () {
        const fileInput = document.getElementById("csvFileInput");
        const output = document.getElementById("output");

        // Clear previous output
        output.innerHTML = "";

        if (fileInput.files.length === 0) {
            output.innerHTML = "Please upload a CSV file.";
            return;
        }

        const file = fileInput.files[0];

        // Check file type
        if (file.type !== "text/csv" && !file.name.endsWith(".csv")) {
            output.innerHTML = "Invalid file type. Please upload a .csv file.";
            return;
        }

        // Prepare the FormData object
        const formData = new FormData();
        formData.append("file", file);

        // Send the file to the Flask backend
        fetch("/upload_csv", {
            method: "POST",
            body: formData,
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error("Failed to upload the file.");
            }
            return response.json();
        })
        .then((data) => {
            if (data.error) {
                output.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            } else {
                output.innerHTML = `<p style="color: green;">Success: ${data.message}</p>`;
                output.innerHTML += `<p>Columns: ${data.columns.join(", ")}</p>`;
            }
        })
        .catch((error) => {
            output.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
        });
    });
});
