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

has 'learning_rate' => (
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

has 'delta_l' => (
    is => 'rw',
    isa => 'ArrayRef[Num]',
    default => sub{ [] }
    );

has 'labels' => (
    is => 'rw',
    isa => 'ArrayRef[Str]',
    required => 1
    );

has 'feature_functions' => (
    is => 'rw',
    isa => 'ArrayRef',
    required => 1
    );

sub BUILD {
    my $self = shift;

    $self->_extract_feature();
    $self->_init_weight();
}

sub _init_weight {
    my $self = shift;
    
    $self->{weight} = [];
    for(my $i = 0; $i < @{ $self->{feature_functions} }; $i++) {
	push @{ $self->{weight} }, 0.0;
    }
}


sub _extract_feature {
    my $self = shift;
    
    $self->{features} = {};
    for(my $doc_i = 0; $doc_i < @{ $self->{docs} }; $doc_i++){
	foreach my $label (@{ $self->{labels} }) {
	    my @vector;
	    foreach my $func (@{ $self->{feature_functions} }){
		push @vector,$func->($self->{docs}->[$doc_i]->{text}, $label);
	    }
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
	foreach my $label (@{ $self->{labels} }) {
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
	$self->_compute_delta();
	$self->_compute_weight();
    }
}

sub _compute_weight {
    my $self = shift;

    my $next_weight = [];
    for(my $vector_i = 0; $vector_i < @{ $self->{feature_functions} }; $vector_i++){
	$next_weight->[$vector_i] += $self->{weight}->[$vector_i] + $self->{learning_rate} * $self->{delta_l}->[$vector_i];
    }
    $self->{weight} = $next_weight;
}

sub _compute_delta {
    my $self = shift;

    $self->{delta_l} = [];
    for(my $doc_i = 0; $doc_i < @{ $self->{docs} }; $doc_i++){
	my $doc_vector = $self->{features}->{$doc_i}->{ $self->{docs}->[$doc_i]->{label} }->{vector};

	for(my $vector_i = 0; $vector_i < @{ $doc_vector }; $vector_i++){
	    $self->{delta_l}->[$vector_i] += $doc_vector->[$vector_i];
	}
	foreach my $label (@{ $self->{labels} }) {
	    for(my $vector_i = 0; $vector_i < @{ $self->{features}->{$doc_i}->{$label}->{vector} }; $vector_i++){
		my $y_given_d = 1.0 / $self->{Z}->[$doc_i] * exp(_dot($self->{weight},$self->{features}->{$doc_i}->{$label}->{vector}));
		$self->{delta_l}->[$vector_i] -= $y_given_d * $self->{features}->{$doc_i}->{$label}->{vector}->[$vector_i];
	    }
	}
    }

    for(my $vector_i = 0; $vector_i < @{ $self->{feature_functions} }; $vector_i++){
	$self->{delta_l}->[$vector_i] -= $self->{C} * $self->{weight}->[$vector_i];
    }
}

sub _compute_Z {
    my $self = shift;

    for(my $doc_i = 0; $doc_i < @{ $self->{docs} }; $doc_i++){
	$self->{Z}->[$doc_i] = 0.0;
	foreach my $label (@{ $self->{labels} }) {
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

__PACKAGE__->meta->make_immutable();

1;
