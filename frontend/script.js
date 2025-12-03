const API_BASE_URL = 'http://localhost:5000';
const SEARCH_ENDPOINT = '/api/search';

const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const resultsContainer = document.getElementById('resultsContainer');

searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

searchButton.addEventListener('click', performSearch);

async function performSearch() {
    const query = searchInput.value.trim();
    
    if (!query) {
        showError('Please enter a search query');
        return;
    }

    showLoading();

    try {
        // const response = await fetch(`${API_BASE_URL}${SEARCH_ENDPOINT}?q=${encodeURIComponent(query)}`);
        // const data = await response.json();
        
        // For now, show a placeholder message
        setTimeout(() => {
            showEmpty('Backend API not yet connected. Results will appear here once the backend is ready.');
        }, 500);

        // if (data.error) {
        //     showError(data.error);
        // } else {
        //     displayResults(data.results || []);
        // }

    } catch (error) {
        console.error('Search error:', error);
        showError('Failed to search. Please try again later.');
    }
}

function showLoading() {
    resultsContainer.innerHTML = `
        <div class="loading">
            <span class="spinner"></span>
            Searching for recipes...
        </div>
    `;
    searchButton.disabled = true;
}

function showError(message) {
    resultsContainer.innerHTML = `
        <div class="error">
            ${message}
        </div>
    `;
    searchButton.disabled = false;
}

function showEmpty(message) {
    resultsContainer.innerHTML = `
        <div class="empty-state">
            ${message}
        </div>
    `;
    searchButton.disabled = false;
}

function displayResults(results) {
    if (results.length === 0) {
        showEmpty('No recipes found. Try a different search term.');
        return;
    }

    const resultsHTML = `
        <div class="results-header">
            Found ${results.length} recipe${results.length !== 1 ? 's' : ''}
        </div>
        <div class="results-grid">
            ${results.map(recipe => createRecipeCard(recipe)).join('')}
        </div>
    `;

    resultsContainer.innerHTML = resultsHTML;
    searchButton.disabled = false;
}

function createRecipeCard(recipe) {
    const ingredients = Array.isArray(recipe.ingredients) 
        ? recipe.ingredients.slice(0, 5).join(', ') + (recipe.ingredients.length > 5 ? '...' : '')
        : (recipe.ingredients || 'N/A');

    const tags = Array.isArray(recipe.tags) ? recipe.tags.slice(0, 5) : [];
    const tagsHTML = tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('');

    const time = recipe.total_time ? `${Math.round(recipe.total_time)} min` : 'N/A';

    const rating = recipe.rating ? `⭐ ${recipe.rating.toFixed(1)}` : 'No rating';

    return `
        <div class="recipe-card">
            <div class="recipe-title">${escapeHtml(recipe.title || 'Untitled Recipe')}</div>
            <div class="recipe-meta">
                <span>⏱️ ${time}</span>
                <span>${rating}</span>
            </div>
            ${tags.length > 0 ? `<div class="recipe-tags">${tagsHTML}</div>` : ''}
            <div class="recipe-ingredients">
                <strong>Ingredients:</strong> ${escapeHtml(ingredients)}
            </div>
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

window.addEventListener('load', () => {
    searchInput.focus();
});
