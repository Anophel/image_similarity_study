const { Pool } = require('pg');

const pool = new Pool({
    user: 'image_similarity_app',
    host: 'localhost',
    database: 'image_similarity_app',
    password: 'Similarity42',
    port: 5432,
});

module.exports = {
    query: (text, params, callback) => {
        return pool.query(text, params, callback)
    },
}
