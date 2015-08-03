use Test::More;

require_ok 'Algorithm::MaximumEntropy';
require_ok 'Algorithm::MaximumEntropy::Feature';
require_ok 'Algorithm::MaximumEntropy::Doc';

my @docs;
push @docs, Algorithm::MaximumEntropy::Doc->new(text => 'good bad good good', label => 'P');
push @docs, Algorithm::MaximumEntropy::Doc->new(text => 'exciting exciting', label => 'P');
push @docs, Algorithm::MaximumEntropy::Doc->new(text => 'bad boring boring boring', label => 'N');
push @docs, Algorithm::MaximumEntropy::Doc->new(text => 'bad exciting bad', label => 'N');

my $me = Algorithm::MaximumEntropy->new(docs => \@docs, size => 8, iter_limit => 1);

$me->train();

my @weight = (0.05, -0.05, 0.00, -0.05, -0.05, 0.05, 0.00, 0.05);
for(my $i = 0; $i < 8; $i++){
    is (sprintf("%.2lf",$me->{weight}->[$i]), sprintf("%.2lf",$weight[$i]));
}

$me = Algorithm::MaximumEntropy->new(docs => \@docs, size => 8, iter_limit => 10000);
$me->train();

@weight = (0.42, -0.25, 0.06, -0.26, -0.42, 0.25, -0.06, 0.26);
for(my $i = 0; $i < 8; $i++){
    is (sprintf("%.2lf",$me->{weight}->[$i]), sprintf("%.2lf",$weight[$i]));
}

done_testing;
