use Test::More;

require_ok 'Algorithm::MaximumEntropy';
require_ok 'Algorithm::MaximumEntropy::Feature';
require_ok 'Algorithm::MaximumEntropy::Doc';

my @docs;
push @docs, Algorithm::MaximumEntropy::Doc->new(text => 'good bad good good', label => 'P');
push @docs, Algorithm::MaximumEntropy::Doc->new(text => 'exciting exciting', label => 'P');
push @docs, Algorithm::MaximumEntropy::Doc->new(text => 'bad boring boring boring', label => 'N');
push @docs, Algorithm::MaximumEntropy::Doc->new(text => 'bad exciting bad', label => 'N');

my @feature_functions;
push @feature_functions, sub {
    my ($doc, $label) = @_;
    return 0 if($label ne 'P');
    return scalar (grep { $_ eq 'good' } split(/ /,$doc)) > 0 ? 1 : 0;
};

push @feature_functions, sub {
    my ($doc, $label) = @_;
    return 0 if($label ne 'P');
    return scalar (grep { $_ eq 'bad' } split(/ /,$doc)) > 0 ? 1 : 0;
};

push @feature_functions, sub {
    my ($doc, $label) = @_;
    return 0 if($label ne 'P');
    return scalar (grep { $_ eq 'exciting' } split(/ /,$doc)) > 0 ? 1 : 0;
};

push @feature_functions, sub {
    my ($doc, $label) = @_;
    return 0 if($label ne 'P');
    return scalar (grep { $_ eq 'boring' } split(/ /,$doc)) > 0 ? 1 : 0;
};

push @feature_functions, sub {
    my ($doc, $label) = @_;
    return 0 if($label ne 'N');
    return scalar (grep { $_ eq 'good' } split(/ /,$doc)) > 0 ? 1 : 0;
};

push @feature_functions, sub {
    my ($doc, $label) = @_;
    return 0 if($label ne 'N');
    return scalar (grep { $_ eq 'bad' } split(/ /,$doc)) > 0 ? 1 : 0;
};

push @feature_functions, sub {
    my ($doc, $label) = @_;
    return 0 if($label ne 'N');
    return scalar (grep { $_ eq 'exciting' } split(/ /,$doc)) > 0 ? 1 : 0;
};

push @feature_functions, sub {
    my ($doc, $label) = @_;
    return 0 if($label ne 'N');
    return scalar (grep { $_ eq 'boring' } split(/ /,$doc)) > 0 ? 1 : 0;
};

my $me = Algorithm::MaximumEntropy->new(docs => \@docs, feature_functions => \@feature_functions, size => 8, iter_limit => 1,labels => ['P','N']);

$me->train();

my @weight = (0.05, -0.05, 0.00, -0.05, -0.05, 0.05, 0.00, 0.05);
for(my $i = 0; $i < 8; $i++){
    is (sprintf("%.2lf",$me->{weight}->[$i]), sprintf("%.2lf",$weight[$i]));
}

$me = Algorithm::MaximumEntropy->new(docs => \@docs, feature_functions => \@feature_functions, size => 8, iter_limit => 20000, labels => ['P','N']);
$me->train();

@weight = (0.42, -0.25, 0.06, -0.26, -0.42, 0.25, -0.06, 0.26);
for(my $i = 0; $i < 8; $i++){
    is (sprintf("%.2lf",$me->{weight}->[$i]), sprintf("%.2lf",$weight[$i]));
}

@docs = ();
push @docs, Algorithm::MaximumEntropy::Doc->new(text => 'exciting boring');
$me->docs(\@docs);
my @result = $me->predict();
is (sprintf("%.2lf",$result[0]->{N}),sprintf("%.2lf",0.60));
is (sprintf("%.2lf",$result[0]->{P}),sprintf("%.2lf",0.40));

done_testing;
