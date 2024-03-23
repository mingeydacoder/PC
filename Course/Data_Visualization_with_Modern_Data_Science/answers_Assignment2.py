answer_01 =\
"""
-- BEGIN SOLUTION
select *
from election_types;

-- END SOLUTION
"""

answer_02 =\
"""
-- BEGIN SOLUTION
select distinct county
from districts;
-- END SOLUTION
"""

answer_03 =\
"""
-- BEGIN SOLUTION
select distinct town
from districts
WHERE COUNTY LIKE "臺南市";
-- END SOLUTION
"""

answer_04 =\
"""
-- BEGIN SOLUTION
SELECT DISTINCT county
  FROM districts
 WHERE county NOT LIKE "臺南市" AND 
       county NOT LIKE "臺北市" AND 
       county NOT LIKE "臺中市" AND 
       county NOT LIKE "高雄市" AND 
       county NOT LIKE "桃園市" AND 
       county NOT LIKE "新北市";
-- END SOLUTION
"""

answer_05 =\
"""
-- BEGIN SOLUTION
SELECT count(polling_place)
FROM polling_places
where election_type_id = 1;
-- END SOLUTION
"""

answer_06 =\
"""
-- BEGIN SOLUTION
SELECT count(*)
FROM candidates
where election_type_id != 1;
-- END SOLUTION
"""

answer_07 =\
"""
-- BEGIN SOLUTION
SELECT party_id, sum(votes)
FROM party_legislators
group by party_id
order by votes desc;
-- END SOLUTION
"""

answer_08 =\
"""
-- BEGIN SOLUTION
SELECT round(sum(votes)*0.05)
FROM party_legislators
;
-- END SOLUTION
"""

answer_09 =\
"""
-- BEGIN SOLUTION
SELECT party_id, name, sum(votes) as sum
FROM party_legislators
inner join parties on parties.id = party_legislators.party_id
group by party_id
having sum > 688873
order by sum desc;
-- END SOLUTION
"""

answer_10 =\
"""
-- BEGIN SOLUTION
select distinct number, parties.name, candidates.name, sum(votes), sum(votes)/(3690466.0+5586019+4671021)
from presidents
inner join candidates on candidates.id = presidents.candidate_id
inner join parties on parties.id = candidates.party_id
group by number;
-- END SOLUTION
"""