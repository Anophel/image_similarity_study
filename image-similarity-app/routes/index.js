var express = require('express');
var router = express.Router();
const db = require('../database/db');

/* GET /annotation
  Grab a random triplet which the user did not see yet.
*/
router.get('/', function (req, res, next) {
  console.log("User " + req.auth.user + " getting new triplet.");

  db.query(`select t.* from triplets t
  left join triplets_annotation ta on t.id = ta.triplet_id 
  and ta.user_id = (select u.id from users u where u.name = $1)
  where ta.id is null
  order by random() limit 1`, [req.auth.user], (err, result) => {
    if (err) {
      console.log(err);
      res.status(500).send("Getting new users did not go well :)");
    }

    res.locals = {
      username: req.auth.user
    };

    if (result.rows.length === 1) {
      console.log(result.rows[0]);
      // Render the page with retrieved triplet
      res.render('index', { triplet: result.rows[0], timestamp: new Date().getTime() });
    } else {
      // Return empty triplet, because user annotated all triplets
      res.render('index', { triplet: null, timestamp: new Date().getTime() });
    }
  });
});

/* POST /annotation
  Saves annotation of the triplet and redirects to the route GET /annotation.
*/
router.post('/', function (req, res, next) {
  const tripletId = Number(req.body.triplet_id);
  const option = Number(req.body.option);
  const timestampStart = Number(req.body.timestamp_start);
  const timeSpent = new Date().getTime() - timestampStart;

  db.query("select * from triplets where id = $1", [tripletId], (errTriplets, resultTriplets) => {
    if (errTriplets || resultTriplets.rows.length !== 1) {
      console.log(errTriplets);
      res.status(500).send("Error in saving annotations");
    }
    const triplet = resultTriplets.rows[0];

    db.query("select * from users where name = $1", [req.auth.user], (errUser, resultUser) => {
      if (errUser || resultUser.rows.length !== 1) {
        console.log(errUser);
        res.status(500).send("Error in retrieving user");
      }
      const user = resultUser.rows[0];

      db.query("insert into triplets_annotation values (nextval('default_sequence'), $1, $2, $3, $4, $5)",
        [triplet.id, user.id, option, option === 1 ? triplet.option_one_path : triplet.option_two_path, timeSpent],
        (errInsert, resultInsert) => {
          if (errInsert) {
            console.log(errInsert);
            res.status(500).send("Error in saving annotation");
          }

          res.redirect("/annotation");
        });
    });
  });

});

module.exports = router;
