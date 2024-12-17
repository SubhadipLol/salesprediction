$(document).ready(function() {
    $('#show-table-btn').click(function() {
        $.ajax({
            url: '/top_predictions',
            method: 'GET',
            success: function(response) {
                $('#table-container').html(response);
            },
            error: function(error) {
                console.log("Error: ", error);
                alert("Failed to load the table.");
            }
        });
    });

    $('#back').click(function() {
        window.location.href = '/menu';
    });
});