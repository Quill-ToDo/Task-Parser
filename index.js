var rr = require('rrule');

function main() {
    let rule = rr.RRule.fromText('8:00 PM every night for the next week');
    console.log(rule.toText());
}

main();