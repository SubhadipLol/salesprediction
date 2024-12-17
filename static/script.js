let csvread_check=false;
let clean_check=false;
let train_check=false;
let menu_check=false;

function handleButtonClick() {
    if (!csvread_check) {
        readCSV();
        document.getElementById("center-button").innerText = "Data cleaning";
    } else if (!clean_check) {
        cleanData();
        document.getElementById("center-button").innerText = "Train the model";
    } 
    else if(!train_check){
        traindata();
        document.getElementById("center-button").innerText = "Proceed to see the results";

    }
    else if(!menu_check){
        window.location.href = '/menu';
        menu_check=true;
    }
    else {
        document.getElementById("status").innerText = "Data has already been trained!";
    }
}
function readCSV() {
    // Change status message to indicate the reading has started
    $('#status').text("Reading CSV file... Please wait.");

    console.log("Button clicked, sending AJAX request to read CSV");

    $.ajax({
        url: "/read_csv",  // The Flask route to read CSV
        type: "GET",
        success: function(response) {
            console.log("AJAX success:", response);
            $('#status').text(response.message);  // Display success message
            csvread_check=true;
        },
        error: function(xhr, status, error) {
            console.log("AJAX error:", error);
            $('#status').text("Error cleaning,try again");
        }
    });
}
function cleanData() {
    $('#status').text("Cleaning the data please wait");

    console.log("Button clicked, sending AJAX request to clean");

    $.ajax({
        url: "/cleaing",  // The Flask route to read CSV
        type: "GET",
        success: function(response) {
            console.log("AJAX success:", response);
            $('#status').text(response.message);  // Display success message
            clean_check=true;
        },
        error: function(xhr, status, error) {
            console.log("AJAX error:", error);
            $('#status').text("Error cleaing the data, try again");
        }
    });
}

function traindata() {
    $('#status').text("Training the data please wait.....");

    console.log("Button clicked, sending AJAX request to train");

    $.ajax({
        url: "/training",  // The Flask route to read CSV
        type: "GET",
        success: function(response) {
            console.log("AJAX success:", response);
            $('#status').text(response.message);  // Display success message
            train_check=true;
        },
        error: function(xhr, status, error) {
            console.log("AJAX error:", error);
            $('#status').text("Error training the data, try again");
        }
    });
}

function navigateToPage() {
    window.location.href = '/choosemodel'; // Redirects to page2.html
}