$(document).ready(function() {
    $('.scrollable-row').each(function() {
        if ($(this).height() > 100) { // Adjust the threshold height as needed
            $(this).addClass('scrollable');
        }
    });
});
