var express = require('express');
var router = express.Router();
const db = require('../database/db');

/* GET explorer page. */
router.get('/', function (req, res, next) {
    if (req.auth.user !== "admin") {
        res.status(403);
        res.send("Unauthorized access.");
    }

    const userInspect = req.query.user;

    db.query(`select ta.id, t.target_path , t.option_one_path , t.option_two_path , ta.choice 
        from triplets_annotation ta 
        join triplets t on ta.triplet_id = t.id
        join users u on ta.user_id = u.id 
        where u."name" = $1
        order by ta.id`, [userInspect], (err, result) => {
        if (err) {
            console.log(err);
            res.status(500).send("Getting new users did not go well :)");
        }

        res.locals = {
            username: req.auth.user
        };

        res.render('explorer', { annotations: result.rows });
    });
});

module.exports = router;
