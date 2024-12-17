document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('radiohead-form');
    const selectedAlbum = document.getElementById('selected-album');

    form.addEventListener('change', () => {
        const selectedRadio = form.querySelector('input[name="album"]:checked');
        
        if (selectedRadio) {
            selectedAlbum.textContent = `You selected: ${selectedRadio.parentElement.textContent.trim()}`;
        }
    });
    
    form.addEventListener('submit', (event) => {
        event.preventDefault(); // Prevent page reload
        
        const selectedRadio = form.querySelector('input[name="album"]:checked');
        const selectedRadioval = selectedRadio.value;

        if (selectedRadioval==1) {
            window.location.href = '/visuals';
        } 
        else if(selectedRadioval==2){
            window.location.href='/top_prediction';
        }
        else if(selectedRadioval==3){
            window.location.href='/export';
        }
        else if(selectedRadioval==4){
            window.location.href='/price_category';
        }
        else if(selectedRadioval==5){
            window.location.href='/'
        }
        else {
            selectedAlbum.textContent = 'No option selected.';
        }
    });
});
