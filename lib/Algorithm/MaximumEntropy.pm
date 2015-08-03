package Algorithm::MaximumEntropy;

use Mouse;
use Algorithm::MaximumEntropy::Feature;

has 'docs' => (
    is => 'rw',
    isa => 'ArrayRef[Algorithm::MaximumEntropy::Doc]'
    );

has 'features' => (
    is => 'rw',
    isa => 'HashRef[HashRef[Algorithm::MaximumEntropy::Feature]]',
    default => sub{ {} }
    );

has 'size' => (
    is => 'rw',
    isa => 'Int'
    );

has 'weight' => (
    is => 'rw',
    isa => 'ArrayRef[Num]',
    default => sub{ [] }
    );

has 'Z' => (
    is => 'rw',
    isa => 'ArrayRef[Num]',
    default => sub{ [] }
    );

has 'alpha' => (
    is => 'rw',
    isa => 'Num',
    default => 0.1
    );

has 'C' => (
    is => 'rw',
    isa => 'Num',
    default => 1.0
    );

has 'iter_limit' => (
    is => 'rw',
    isa => 'Int',
    default => 100
    );

sub BUILD {
    my $self = shift;

    $self->_extract_feature();
    $self->_init_weight();
}

sub _init_weight {
    my $self = shift;
    
    $self->{weight} = [];
    for(my $i = 0; $i < $self->size; $i++) {
	push @{ $self->{weight} }, 0.0;
    }
}


sub _extract_feature {
    my $self = shift;
    
    $self->{features} = {};
    for(my $doc_i = 0; $doc_i < @{ $self->{docs} }; $doc_i++){
	foreach my $label ('P','N') {
	    my @vector;
	    push @vector, _feature_func1($self->{docs}->[$doc_i]->{text}, $label);
	    push @vector, _feature_func2($self->{docs}->[$doc_i]->{text}, $label);
	    push @vector, _feature_func3($self->{docs}->[$doc_i]->{text}, $label);
	    push @vector, _feature_func4($self->{docs}->[$doc_i]->{text}, $label);
	    push @vector, _feature_func5($self->{docs}->[$doc_i]->{text}, $label);
	    push @vector, _feature_func6($self->{docs}->[$doc_i]->{text}, $label);
	    push @vector, _feature_func7($self->{docs}->[$doc_i]->{text}, $label);
	    push @vector, _feature_func8($self->{docs}->[$doc_i]->{text}, $label);

	    $self->{features}->{$doc_i}->{$label}
	    = Algorithm::MaximumEntropy::Feature->new(vector => \@vector,label => $label);
	}
    }
};

sub predict {
    my $self = shift;

    $self->_extract_feature();
    $self->_compute_Z();

    my @all_result;
    for(my $doc_i = 0; $doc_i < @{ $self->{docs} }; $doc_i++){
	my $hash;
	foreach my $label ('P','N') {
	    my $y_given_d = 1.0 / $self->{Z}->[$doc_i] * exp(_dot($self->{weight},$self->{features}->{$doc_i}->{$label}->{vector}));
	    $hash->{$label} = $y_given_d;
	}
	push @all_result, $hash;
    }
    return @all_result;
}

sub train {
    my $self = shift;

    for(my $round = 0; $round < $self->{iter_limit}; $round++){
	$self->_compute_Z();
	my $delta_l = $self->_compute_delta();
	$self->_compute_weight($delta_l);
    }
}

sub _compute_weight {
    my ($self,$delta_l) = @_;

    my $next_weight = [];
    for(my $vector_i = 0; $vector_i < $self->{size}; $vector_i++){
	$next_weight->[$vector_i] += $self->{weight}->[$vector_i] + $self->{alpha} * $delta_l->[$vector_i];
    }
    $self->{weight} = $next_weight;
}

sub _compute_delta {
    my $self = shift;

    my $delta_l = [];
    for(my $doc_i = 0; $doc_i < @{ $self->{docs} }; $doc_i++){
	my $doc_vector = $self->{features}->{$doc_i}->{ $self->{docs}->[$doc_i]->{label} }->{vector};

	for(my $vector_i = 0; $vector_i < @{ $doc_vector }; $vector_i++){
	    $delta_l->[$vector_i] += $doc_vector->[$vector_i];
	}
	foreach my $label ('P','N') {
	    for(my $vector_i = 0; $vector_i < @{ $self->{features}->{$doc_i}->{$label}->{vector} }; $vector_i++){
		my $y_given_d = 1.0 / $self->{Z}->[$doc_i] * exp(_dot($self->{weight},$self->{features}->{$doc_i}->{$label}->{vector}));
		$delta_l->[$vector_i] -= $y_given_d * $self->{features}->{$doc_i}->{$label}->{vector}->[$vector_i];
	    }
	}
    }

    for(my $vector_i = 0; $vector_i < $self->{size}; $vector_i++){
	$delta_l->[$vector_i] -= $self->{C} * $self->{weight}->[$vector_i];
    }
    return $delta_l;
}

sub _compute_Z {
    my $self = shift;

    for(my $doc_i = 0; $doc_i < @{ $self->{docs} }; $doc_i++){
	$self->{Z}->[$doc_i] = 0.0;
	foreach my $label ('P','N') {
	    my $sum = 0.0;
	    for(my $vector_i = 0; $vector_i < @{ $self->{features}->{$doc_i}->{$label}->{vector} }; $vector_i++){
		$sum += $self->{weight}->[$vector_i]
		    * $self->{features}->{$doc_i}->{$label}->{vector}->[$vector_i];
	    }
	    $self->{Z}->[$doc_i] += exp($sum);
	}
    }
}

sub _dot {
    my ($vector1, $vector2) = @_;

    my $sum = 0;
    for(my $i = 0; $i < @{ $vector1 }; $i++){
	$sum += $vector1->[$i] * $vector2->[$i];
    }
    return $sum;
}

sub _feature_func1 {
    my ($doc, $label) = @_;
    return 0 if($label ne 'P');
    return scalar (grep { $_ eq 'good' } split(/ /,$doc)) > 0 ? 1 : 0;
}

sub _feature_func2 {
    my ($doc, $label) = @_;
    return 0 if($label ne 'P');
    return scalar (grep { $_ eq 'bad' } split(/ /,$doc)) > 0 ? 1 : 0;
}

sub _feature_func3 {
    my ($doc, $label) = @_;
    return 0 if($label ne 'P');
    return scalar (grep { $_ eq 'exciting' } split(/ /,$doc)) > 0 ? 1 : 0;
}

sub _feature_func4 {
    my ($doc, $label) = @_;
    return 0 if($label ne 'P');
    return scalar (grep { $_ eq 'boring' } split(/ /,$doc)) > 0 ? 1 : 0;
}

sub _feature_func5 {
    my ($doc, $label) = @_;
    return 0 if($label ne 'N');
    return scalar (grep { $_ eq 'good' } split(/ /,$doc)) > 0 ? 1 : 0;
}

sub _feature_func6 {
    my ($doc, $label) = @_;
    return 0 if($label ne 'N');
    return scalar (grep { $_ eq 'bad' } split(/ /,$doc)) > 0 ? 1 : 0;
}

sub _feature_func7 {
    my ($doc, $label) = @_;
    return 0 if($label ne 'N');
    return scalar (grep { $_ eq 'exciting' } split(/ /,$doc)) > 0 ? 1 : 0;
}

sub _feature_func8 {
    my ($doc, $label) = @_;
    return 0 if($label ne 'N');
    return scalar (grep { $_ eq 'boring' } split(/ /,$doc)) > 0 ? 1 : 0;
}

__PACKAGE__->meta->make_immutable();

1;
